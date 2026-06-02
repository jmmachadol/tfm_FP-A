"""Modelos de Machine Learning y Deep Learning: LightGBM, MLP y N-BEATS.

Los tres siguen el paradigma de *global forecasting*: un único modelo se
entrena sobre todas las series de la submuestra (o de cada fold) de forma
simultánea, aprendiendo patrones compartidos. Implementan ``GlobalBaseForecaster``.

Las características de entrada (lags) se escalan por serie con MaxAbs antes de
construir la matriz tabular global; al predecir, se aplica el mismo escalado
a la ventana de inferencia de la serie en cuestión.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np

from src.models.base import GlobalBaseForecaster

# ── LightGBM ──────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ── PyTorch ───────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_EPS = 1e-8


def _build_lag_matrix(
    series: np.ndarray, n_lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construye matrices de características (lags) y objetivo multi-paso.

    Para cada posición t válida extrae los últimos ``n_lags`` valores como
    features y los siguientes ``horizon`` valores como target.

    Returns:
        Tupla (X, y) con shapes (n_samples, n_lags) y (n_samples, horizon).
    """
    n = len(series)
    if n < n_lags + horizon:
        return np.empty((0, n_lags)), np.empty((0, horizon))
    X, y = [], []
    for t in range(n - n_lags - horizon + 1):
        X.append(series[t : t + n_lags])
        y.append(series[t + n_lags : t + n_lags + horizon])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _scale_series(series: np.ndarray) -> tuple[np.ndarray, float]:
    """MaxAbs scaling: divide por el máximo del valor absoluto."""
    scale = float(np.max(np.abs(series)))
    if scale < _EPS:
        scale = 1.0
    return series / scale, scale


# ─────────────────────────────────────────────────────────────────────────────
#  LightGBM
# ─────────────────────────────────────────────────────────────────────────────
class LightGBMForecaster(GlobalBaseForecaster):
    """Gradient boosting tabular con LightGBM y predicción recursiva.

    Construye una matriz de features global combinando series de múltiples
    series (escaladas por MaxAbs independientemente). El modelo predice
    un paso a la vez y alimenta la predicción anterior al paso siguiente
    (recursive forecasting).

    Args:
        n_lags: Número de rezagos usados como features.
        n_estimators: Rondas de boosting.
        learning_rate: Tasa de aprendizaje.
        num_leaves: Número máximo de hojas por árbol.
        seed: Semilla de aleatoriedad.
    """

    def __init__(
        self,
        n_lags: int = 36,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        seed: int = 42,
    ) -> None:
        if not _HAS_LGB:
            raise ImportError("lightgbm no está instalado.")
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self._model: lgb.Booster | None = None

    def fit_global(self, train_series: List[np.ndarray]) -> "LightGBMForecaster":
        X_all, y_all = [], []
        for s in train_series:
            scaled, _ = _scale_series(np.asarray(s, dtype=float).ravel())
            # Direct forecasting: predict n_lags -> 1 step at a time
            # We use a 1-step-ahead target for recursive prediction
            for t in range(len(scaled) - self.n_lags):
                X_all.append(scaled[t : t + self.n_lags])
                y_all.append(scaled[t + self.n_lags])

        if not X_all:
            raise ValueError("No hay suficientes datos para construir la matriz global.")

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)

        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": self.seed,
        }
        ds = lgb.Dataset(X, label=y)
        self._model = lgb.train(params, ds, num_boost_round=self.n_estimators)
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Llame a fit_global() antes de predict().")
        history = np.asarray(history, dtype=float).ravel()
        scale = float(np.max(np.abs(history[-self.n_lags :])))
        if scale < _EPS:
            scale = 1.0

        window = (history[-self.n_lags :] / scale).tolist()
        preds = []
        for _ in range(horizon):
            x = np.array(window[-self.n_lags :], dtype=np.float32).reshape(1, -1)
            p = float(self._model.predict(x)[0])
            preds.append(p)
            window.append(p)

        return np.array(preds, dtype=float) * scale


# ─────────────────────────────────────────────────────────────────────────────
#  MLP Bottleneck (Direct Forecasting)
# ─────────────────────────────────────────────────────────────────────────────
class _MLPBottleneck(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPForecaster(GlobalBaseForecaster):
    """MLP bottleneck con direct forecasting multi-horizonte (PyTorch).

    Predice los ``horizon`` pasos futuros en una sola pasada forward,
    eliminando el error acumulativo de la predicción recursiva.

    Args:
        n_lags: Ventana de entrada.
        horizon: Número de pasos futuros predichos de una vez.
        epochs: Épocas de entrenamiento.
        batch_size: Tamaño de mini-batch.
        lr: Tasa de aprendizaje inicial (AdamW).
        seed: Semilla de aleatoriedad.
    """

    def __init__(
        self,
        n_lags: int = 36,
        horizon: int = 18,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 0.002,
        seed: int = 42,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch no está instalado.")
        self.n_lags = n_lags
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self._net: _MLPBottleneck | None = None

    def fit_global(self, train_series: List[np.ndarray]) -> "MLPForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_all, y_all = [], []
        for s in train_series:
            scaled, _ = _scale_series(np.asarray(s, dtype=float).ravel())
            Xs, ys = _build_lag_matrix(scaled, self.n_lags, self.horizon)
            if Xs.shape[0] > 0:
                X_all.append(Xs)
                y_all.append(ys)

        if not X_all:
            raise ValueError("No hay datos suficientes para entrenar el MLP.")

        X = torch.tensor(np.vstack(X_all), dtype=torch.float32)
        y = torch.tensor(np.vstack(y_all), dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X, y), batch_size=self.batch_size, shuffle=True
        )
        self._net = _MLPBottleneck(self.n_lags, self.horizon)
        optimizer = optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.L1Loss()

        self._net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                if bx.shape[0] < 2:  # BatchNorm requiere batch >= 2
                    continue
                optimizer.zero_grad()
                criterion(self._net(bx), by).backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Llame a fit_global() antes de predict().")
        history = np.asarray(history, dtype=float).ravel()
        scale = float(np.max(np.abs(history[-self.n_lags :])))
        if scale < _EPS:
            scale = 1.0
        x = torch.tensor(
            (history[-self.n_lags :] / scale), dtype=torch.float32
        ).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(x).numpy()[0]
        return np.maximum(pred, 0.0) * scale


# ─────────────────────────────────────────────────────────────────────────────
#  N-BEATS Interpretable (Direct Forecasting)
# ─────────────────────────────────────────────────────────────────────────────
class _TrendBasis(nn.Module):
    """Base polinomial para el stack de tendencia de N-BEATS."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int) -> None:
        super().__init__()
        self.degree = degree
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        # Vectores de tiempo normalizados [-1, 1]
        b_time = torch.linspace(-1, 1, backcast_size)
        f_time = torch.linspace(0, 1, forecast_size)
        # Bases de Chebyshev hasta grado `degree`
        B_b = torch.stack([b_time**d for d in range(degree + 1)], dim=1)
        B_f = torch.stack([f_time**d for d in range(degree + 1)], dim=1)
        self.register_buffer("B_b", B_b)  # (backcast_size, degree+1)
        self.register_buffer("B_f", B_f)  # (forecast_size, degree+1)

    def forward(self, theta: torch.Tensor):
        # theta: (batch, 2*(degree+1))
        d = self.degree + 1
        theta_b = theta[:, :d]   # (batch, d)
        theta_f = theta[:, d:]   # (batch, d)
        backcast = theta_b @ self.B_b.T   # (batch, backcast_size)
        forecast = theta_f @ self.B_f.T   # (batch, forecast_size)
        return backcast, forecast


class _SeasonalityBasis(nn.Module):
    """Base de Fourier para el stack de estacionalidad de N-BEATS."""

    def __init__(
        self, harmonics: int, backcast_size: int, forecast_size: int
    ) -> None:
        super().__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        b_time = torch.linspace(0, 1, backcast_size)
        f_time = torch.linspace(1, 1 + forecast_size / backcast_size, forecast_size)

        freqs = torch.arange(1, harmonics + 1).float()
        B_b_cos = torch.cos(2 * np.pi * b_time.unsqueeze(1) * freqs.unsqueeze(0))
        B_b_sin = torch.sin(2 * np.pi * b_time.unsqueeze(1) * freqs.unsqueeze(0))
        B_b = torch.cat([B_b_cos, B_b_sin], dim=1)  # (backcast, 2*harmonics)

        B_f_cos = torch.cos(2 * np.pi * f_time.unsqueeze(1) * freqs.unsqueeze(0))
        B_f_sin = torch.sin(2 * np.pi * f_time.unsqueeze(1) * freqs.unsqueeze(0))
        B_f = torch.cat([B_f_cos, B_f_sin], dim=1)  # (forecast, 2*harmonics)

        self.register_buffer("B_b", B_b)
        self.register_buffer("B_f", B_f)

    def forward(self, theta: torch.Tensor):
        d = 2 * self.harmonics
        theta_b = theta[:, :d]
        theta_f = theta[:, d:]
        backcast = theta_b @ self.B_b.T
        forecast = theta_f @ self.B_f.T
        return backcast, forecast


class _NBEATSBlock(nn.Module):
    """Bloque residual de N-BEATS con base de funciones configurable."""

    def __init__(
        self, input_size: int, theta_size: int, layer_width: int, n_layers: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_size, layer_width), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(layer_width, layer_width), nn.ReLU()]
        layers.append(nn.Linear(layer_width, theta_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _NBEATSInterpretable(nn.Module):
    """N-BEATS interpretable con dos stacks: tendencia y estacionalidad."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        trend_degree: int = 3,
        seasonality_harmonics: int = 6,
        n_blocks_per_stack: int = 3,
        layer_width: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        trend_theta = 2 * (trend_degree + 1)
        season_theta = 2 * 2 * seasonality_harmonics  # cos+sin, backcast+forecast

        # Stack tendencia
        self.trend_blocks = nn.ModuleList(
            [_NBEATSBlock(input_size, trend_theta, layer_width, n_layers)
             for _ in range(n_blocks_per_stack)]
        )
        self.trend_basis = _TrendBasis(trend_degree, input_size, output_size)

        # Stack estacionalidad
        self.season_blocks = nn.ModuleList(
            [_NBEATSBlock(input_size, season_theta, layer_width, n_layers)
             for _ in range(n_blocks_per_stack)]
        )
        self.season_basis = _SeasonalityBasis(seasonality_harmonics, input_size, output_size)

    def forward(self, x: torch.Tensor):
        residual = x
        forecast = torch.zeros(x.shape[0], self.output_size, device=x.device)

        # Stack tendencia
        for block in self.trend_blocks:
            theta = block(residual)
            backcast, fc = self.trend_basis(theta)
            residual = residual - backcast
            forecast = forecast + fc

        # Stack estacionalidad
        for block in self.season_blocks:
            theta = block(residual)
            backcast, fc = self.season_basis(theta)
            residual = residual - backcast
            forecast = forecast + fc

        return forecast


class NBEATSForecaster(GlobalBaseForecaster):
    """N-BEATS Interpretable global con direct forecasting.

    Implementación en PyTorch puro (sin dependencia de Darts). Entrena un único
    modelo sobre todas las series de la submuestra. Los bloques de tendencia
    emplean bases polinomiales y los de estacionalidad bases de Fourier, lo que
    aporta interpretabilidad estructural alineada con las necesidades de FP&A
    (Oreshkin et al., 2020).

    Args:
        n_lags: Ventana de contexto (input_chunk_length).
        horizon: Horizonte de predicción (output_chunk_length).
        epochs: Épocas de entrenamiento.
        batch_size: Tamaño de mini-batch.
        lr: Tasa de aprendizaje.
        n_blocks: Bloques por stack (tendencia + estacionalidad).
        layer_width: Neuronas por capa oculta.
        seed: Semilla de aleatoriedad.
    """

    def __init__(
        self,
        n_lags: int = 36,
        horizon: int = 18,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
        n_blocks: int = 3,
        layer_width: int = 128,
        seed: int = 42,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch no está instalado.")
        self.n_lags = n_lags
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_blocks = n_blocks
        self.layer_width = layer_width
        self.seed = seed
        self._net: _NBEATSInterpretable | None = None

    def fit_global(self, train_series: List[np.ndarray]) -> "NBEATSForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_all, y_all = [], []
        for s in train_series:
            scaled, _ = _scale_series(np.asarray(s, dtype=float).ravel())
            Xs, ys = _build_lag_matrix(scaled, self.n_lags, self.horizon)
            if Xs.shape[0] > 0:
                X_all.append(Xs)
                y_all.append(ys)

        if not X_all:
            raise ValueError("No hay datos suficientes para entrenar N-BEATS.")

        X = torch.tensor(np.vstack(X_all), dtype=torch.float32)
        y = torch.tensor(np.vstack(y_all), dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X, y), batch_size=self.batch_size, shuffle=True
        )

        self._net = _NBEATSInterpretable(
            input_size=self.n_lags,
            output_size=self.horizon,
            n_blocks_per_stack=self.n_blocks,
            layer_width=self.layer_width,
        )
        optimizer = optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.L1Loss()

        self._net.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ep in range(self.epochs):
                total = 0.0
                for bx, by in loader:
                    if bx.shape[0] < 2:
                        continue
                    optimizer.zero_grad()
                    loss = criterion(self._net(bx), by)
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Llame a fit_global() antes de predict().")
        history = np.asarray(history, dtype=float).ravel()
        scale = float(np.max(np.abs(history[-self.n_lags :])))
        if scale < _EPS:
            scale = 1.0
        x = torch.tensor(
            (history[-self.n_lags :] / scale), dtype=torch.float32
        ).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(x).numpy()[0]
        return np.maximum(pred, 0.0) * scale

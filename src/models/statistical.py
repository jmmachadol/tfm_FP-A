"""Modelos estadísticos clásicos: ETS, Holt-Winters y SARIMA.

- ETS y AutoARIMA vía ``statsforecast`` (AutoETS / AutoARIMA con selección
  automática de componentes por AICc, algoritmo Hyndman-Khandakar).
- Holt-Winters vía ``statsmodels`` (ExponentialSmoothing), con selección
  automática entre variante aditiva y multiplicativa.

Todos implementan la interfaz ``BaseForecaster`` (ajuste local, serie a serie).
Si una variante de Holt-Winters diverge, se prueba la siguiente en orden de
preferencia: multiplicativa → aditiva → solo tendencia → solo nivel.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.models.base import BaseForecaster

# ── statsforecast ────────────────────────────────────────────────────────────
try:
    from statsforecast.models import AutoARIMA as _AutoARIMA
    from statsforecast.models import AutoETS as _AutoETS
    _HAS_STATSFORECAST = True
except ImportError:  # pragma: no cover
    _HAS_STATSFORECAST = False

# ── statsmodels ──────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ES
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

_EPS = 1e-8


class ETSForecaster(BaseForecaster):
    """ETS con selección automática de componentes por AICc (statsforecast).

    Args:
        seasonality: Periodo estacional (12 para datos mensuales).
        error: Tipo de error ('A', 'M', 'Z'=automático).
        allow_multiplicative_trend: Si True, considera tendencias multiplicativas.
    """

    def __init__(self, seasonality: int = 12) -> None:
        if not _HAS_STATSFORECAST:
            raise ImportError("statsforecast no está instalado. Ejecute: pip install statsforecast")
        self.seasonality = seasonality
        self._model: _AutoETS | None = None
        self._fitted = False

    def fit(self, train: np.ndarray) -> "ETSForecaster":
        train = np.asarray(train, dtype=float).ravel()
        self._model = _AutoETS(season_length=self.seasonality)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = self._model.fit(y=train)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llame a fit() antes de predict().")
        result = self._model.predict(h=horizon)
        # statsforecast devuelve dict con clave 'mean'
        pred = result.get("mean", result) if isinstance(result, dict) else result
        return np.asarray(pred, dtype=float).ravel()[:horizon]


class HoltWintersForecaster(BaseForecaster):
    """Holt-Winters con selección automática entre variante aditiva y multiplicativa.

    Prueba en orden: multiplicativo completo → aditivo completo → sin estacionalidad
    multiplicativa → sin estacionalidad aditiva. Usa el modelo que converge primero.

    Args:
        seasonality: Periodo estacional.
        min_periods: Mínimo de observaciones requeridas (al menos 2 ciclos).
    """

    def __init__(self, seasonality: int = 12, min_periods: int = 24) -> None:
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels no está instalado.")
        self.seasonality = seasonality
        self.min_periods = min_periods
        self._fitted_model = None
        self._train: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> "HoltWintersForecaster":
        train = np.asarray(train, dtype=float).ravel()
        if train.size < self.min_periods:
            raise ValueError(
                f"Holt-Winters requiere al menos {self.min_periods} observaciones "
                f"(recibido {train.size})."
            )

        # Si hay valores no positivos la variante multiplicativa no funciona.
        all_positive = bool(np.all(train > _EPS))

        configs = []
        if all_positive:
            configs += [("mul", "mul"), ("add", "mul")]
        configs += [("add", "add"), ("add", None), (None, None)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for trend, seasonal in configs:
                try:
                    sp = self.seasonality if seasonal is not None else None
                    model = _ES(
                        train,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=sp,
                    )
                    fitted = model.fit(optimized=True)
                    if np.isfinite(fitted.aic):
                        self._fitted_model = fitted
                        return self
                except Exception:
                    continue

        raise RuntimeError("No se pudo ajustar ninguna variante de Holt-Winters.")

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted_model is None:
            raise RuntimeError("Llame a fit() antes de predict().")
        return np.asarray(self._fitted_model.forecast(horizon), dtype=float).ravel()


class SARIMAForecaster(BaseForecaster):
    """SARIMA con selección automática de orden mediante AutoARIMA (statsforecast).

    Utiliza el algoritmo de Hyndman-Khandakar para seleccionar (p,d,q)(P,D,Q,s)
    por minimización del AICc. Es el equivalente de ``auto.arima`` de R y de
    ``pmdarima.auto_arima`` en Python, pero significativamente más rápido al
    estar implementado en Cython/C++ dentro de statsforecast.

    Args:
        seasonality: Periodo estacional.
        approximation: Si True, usa aproximaciones que aceleran la búsqueda
            (recomendado para series largas o cuando se ejecutan muchos ajustes).
    """

    def __init__(self, seasonality: int = 12, approximation: bool = True) -> None:
        if not _HAS_STATSFORECAST:
            raise ImportError("statsforecast no está instalado.")
        self.seasonality = seasonality
        self.approximation = approximation
        self._model: _AutoARIMA | None = None
        self._fitted = False

    def fit(self, train: np.ndarray) -> "SARIMAForecaster":
        train = np.asarray(train, dtype=float).ravel()
        self._model = _AutoARIMA(
            season_length=self.seasonality,
            approximation=self.approximation,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = self._model.fit(y=train)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llame a fit() antes de predict().")
        result = self._model.predict(h=horizon)
        pred = result.get("mean", result) if isinstance(result, dict) else result
        return np.asarray(pred, dtype=float).ravel()[:horizon]

"""Modelo baseline: Seasonal Naïve.

El Seasonal Naïve es la referencia mínima para series temporales con
estacionalidad. Pronostica repitiendo el patrón del último ciclo estacional
completo conocido. Cualquier modelo de mayor complejidad debe superarlo para
justificar su uso (Hyndman & Athanasopoulos, 2021).
"""

from __future__ import annotations

import numpy as np

from src.models.base import BaseForecaster


class SeasonalNaive(BaseForecaster):
    """Pronóstico naïve estacional.

    Para datos mensuales (``seasonality=12``), el pronóstico del mes t+h es
    el valor observado en t+h-12 (o t+h-24 si h > 12, etc.), replicando el
    último ciclo anual completo disponible en la historia de entrenamiento.

    Args:
        seasonality: Periodo estacional. Por defecto 12 para datos mensuales.
    """

    def __init__(self, seasonality: int = 12) -> None:
        self.seasonality = seasonality
        self._last_season: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> "SeasonalNaive":
        train = np.asarray(train, dtype=float).ravel()
        if train.size < self.seasonality:
            raise ValueError(
                f"La serie de entrenamiento ({train.size} obs.) debe tener al menos "
                f"{self.seasonality} observaciones para el Seasonal Naïve."
            )
        self._last_season = train[-self.seasonality :]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last_season is None:
            raise RuntimeError("Llame a fit() antes de predict().")
        reps = int(np.ceil(horizon / self.seasonality))
        return np.tile(self._last_season, reps)[:horizon]

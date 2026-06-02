"""Interfaces abstractas para todos los modelos de pronóstico del TFM.

Define dos contratos:
- ``BaseForecaster``: modelos locales (se ajustan serie por serie).
- ``GlobalBaseForecaster``: modelos globales (se ajustan sobre todas las series
  de un fold a la vez y predicen por serie).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseForecaster(ABC):
    """Interfaz común para modelos de ajuste local (una serie a la vez)."""

    IS_GLOBAL: bool = False

    @abstractmethod
    def fit(self, train: np.ndarray) -> "BaseForecaster":
        """Ajusta el modelo sobre la serie de entrenamiento.

        Args:
            train: Valores históricos de la serie, array 1-D.

        Returns:
            El propio objeto ajustado (permite encadenamiento ``fit().predict()``).
        """

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Genera pronósticos para los próximos ``horizon`` pasos.

        Args:
            horizon: Número de pasos futuros a pronosticar.

        Returns:
            Array 1-D de longitud ``horizon`` con los valores pronosticados.
        """

    def fit_predict(self, train: np.ndarray, horizon: int) -> np.ndarray:
        """Atajo que combina ``fit`` y ``predict``."""
        return self.fit(train).predict(horizon)


class GlobalBaseForecaster(ABC):
    """Interfaz para modelos de ajuste global (todas las series de un fold).

    En el paradigma de *global forecasting*, un único modelo se entrena
    simultáneamente sobre múltiples series, aprendiendo patrones compartidos
    entre ellas. El ajuste recibe la lista completa de series de entrenamiento
    disponibles para el fold actual; la predicción se realiza luego sobre cada
    serie de forma individual.
    """

    IS_GLOBAL: bool = True

    @abstractmethod
    def fit_global(self, train_series: List[np.ndarray]) -> "GlobalBaseForecaster":
        """Ajusta el modelo sobre todas las series de entrenamiento del fold.

        Args:
            train_series: Lista de arrays 1-D, una por serie, con los valores
                históricos disponibles hasta el corte del fold actual.

        Returns:
            El propio objeto ajustado.
        """

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Genera pronósticos para una serie dado su historial reciente.

        Args:
            history: Valores históricos de la serie (al menos tantos como los
                rezagos utilizados por el modelo).
            horizon: Número de pasos futuros a pronosticar.

        Returns:
            Array 1-D de longitud ``horizon`` con los valores pronosticados.
        """

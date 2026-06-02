"""Configuración central del experimento.

Reúne en un único lugar las semillas, las rutas relativas y los parámetros del
protocolo experimental, de modo que el resto de módulos no contenga constantes
dispersas. Todas las rutas se derivan de la raíz del proyecto y son relativas,
lo que garantiza la portabilidad entre máquinas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto (relativas a la raíz, derivadas de la ubicación de este
# archivo: src/config.py -> la raíz es el directorio padre de src/).
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

RESULTS_DIR: Path = ROOT_DIR / "results"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# Semilla global única para todo el pipeline (NumPy, PyTorch, LightGBM, modelos).
SEED: int = 42

# URLs oficiales del repositorio de la competición M4 (datos de origen público).
M4_INFO_URL: str = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
    "Dataset/M4-info.csv"
)
M4_MONTHLY_TRAIN_URL: str = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
    "Dataset/Train/Monthly-train.csv"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Parámetros del diseño experimental de la comparativa.

    Attributes:
        seasonality: Periodo estacional. 12 para datos mensuales.
        horizon: Horizonte de pronóstico por iteración, en meses
            (estándar M4 mensual = 18).
        min_length: Longitud mínima de serie para ser incluida, en meses.
        initial_train: Tamaño mínimo de la ventana inicial de entrenamiento
            del walk-forward, en meses.
        step: Desplazamiento entre iteraciones consecutivas del walk-forward,
            en meses.
        window: Tipo de ventana del walk-forward, 'expanding' o 'rolling'.
        n_series: Tamaño de la submuestra estratificada de series. None usa
            todas las series disponibles tras el filtrado.
        category: Categoría M4 a utilizar.
        frequency: Frecuencia M4 a utilizar.
    """

    seasonality: int = 12
    horizon: int = 18
    min_length: int = 72
    initial_train: int = 54
    step: int = 12
    window: str = "expanding"
    n_series: int | None = 400
    category: str = "Finance"
    frequency: str = "Monthly"


CONFIG = ExperimentConfig()

# Nombres canónicos de los modelos de la comparativa (orden de complejidad
# creciente). El orquestador y las tablas de resultados se apoyan en esta lista.
MODELS: tuple[str, ...] = (
    "SNaive",
    "ETS",
    "HoltWinters",
    "SARIMA",
    "LightGBM",
    "MLP",
    "NBEATS",
)

# Métricas reportadas (estadísticas + impacto económico).
METRICS: tuple[str, ...] = ("MAE", "RMSE", "MAPE", "sMAPE", "MASE", "WAPE")

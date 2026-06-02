"""Preprocesamiento: filtrado, submuestra estratificada y escalado.

Transforma el DataFrame M4 en una lista de arrays numpy listos para el
backtesting. Aplica el filtro de longitud mínima y extrae una submuestra
estratificada por cuartil de longitud para garantizar representatividad.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config import CONFIG, ExperimentConfig
from src.utils import get_logger, set_global_seed

_LOG = get_logger("preprocessor")


def compute_lengths(data_df: pd.DataFrame) -> pd.Series:
    """Calcula la longitud real de cada serie (sin NaNs finales)."""
    return data_df.notna().sum(axis=1)


def filter_by_min_length(
    info_df: pd.DataFrame,
    data_df: pd.DataFrame,
    config: ExperimentConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Retiene solo las series con longitud ≥ ``config.min_length``.

    Returns:
        Tupla (info_filtrado, data_filtrado, lengths_filtrado).
    """
    lengths = compute_lengths(data_df)
    info_df = info_df.copy()
    info_df["Actual_Length"] = lengths

    mask = lengths >= config.min_length
    info_filt = info_df[mask].copy()
    data_filt = data_df[mask].copy()
    lengths_filt = lengths[mask]

    _LOG.info(
        "Filtrado ≥%d meses: %d → %d series (%.1f%%)",
        config.min_length,
        len(data_df),
        len(data_filt),
        100.0 * len(data_filt) / len(data_df),
    )
    return info_filt, data_filt, lengths_filt


def stratified_subsample(
    info_df: pd.DataFrame,
    data_df: pd.DataFrame,
    lengths: pd.Series,
    config: ExperimentConfig = CONFIG,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Submuestra estratificada por cuartil de longitud de serie.

    Si ``config.n_series`` es None, devuelve todas las series disponibles.
    La estratificación garantiza que la submuestra representa la distribución
    completa de longitudes del dataset filtrado.

    Args:
        info_df: Metadatos de las series filtradas.
        data_df: Datos de las series filtradas.
        lengths: Longitudes reales de las series filtradas.
        config: Configuración del experimento.
        seed: Semilla de aleatoriedad para reproducibilidad.

    Returns:
        Tupla (info_sub, data_sub).
    """
    if config.n_series is None:
        return info_df, data_df

    set_global_seed(seed)
    n = config.n_series
    quartiles = pd.qcut(lengths, q=4, labels=False, duplicates="drop")
    n_per_stratum = max(1, n // 4)

    sampled_ids = []
    for q in sorted(quartiles.unique()):
        stratum_ids = lengths[quartiles == q].index.tolist()
        k = min(n_per_stratum, len(stratum_ids))
        sampled_ids.extend(
            np.random.choice(stratum_ids, size=k, replace=False).tolist()
        )

    # Ajustar al tamaño exacto n
    if len(sampled_ids) < n:
        remaining = [i for i in data_df.index if i not in set(sampled_ids)]
        extra = min(n - len(sampled_ids), len(remaining))
        sampled_ids.extend(
            np.random.choice(remaining, size=extra, replace=False).tolist()
        )
    elif len(sampled_ids) > n:
        np.random.shuffle(sampled_ids)
        sampled_ids = sampled_ids[:n]

    info_sub = info_df.loc[sampled_ids]
    data_sub = data_df.loc[sampled_ids]
    _LOG.info("Submuestra estratificada: %d series", len(data_sub))
    return info_sub, data_sub


def extract_series_list(data_df: pd.DataFrame) -> List[np.ndarray]:
    """Convierte el DataFrame en una lista de arrays 1-D sin NaNs.

    Returns:
        Lista de arrays numpy, uno por serie, con sus valores reales.
    """
    series = []
    for _, row in data_df.iterrows():
        s = row.dropna().values.astype(float)
        series.append(s)
    return series


def get_volume_deciles(series_list: List[np.ndarray]) -> np.ndarray:
    """Asigna un decil de volumen (0-9) a cada serie según su media absoluta."""
    means = np.array([np.mean(np.abs(s)) for s in series_list])
    deciles = pd.qcut(means, q=10, labels=False, duplicates="drop")
    # pd.qcut puede retornar Categorical, Series o ndarray según versión
    if hasattr(deciles, "to_numpy"):
        return deciles.to_numpy()
    return np.asarray(deciles)

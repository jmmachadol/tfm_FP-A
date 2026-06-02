"""Descarga y caché del dataset M4 Financial Monthly.

Descarga desde el repositorio oficial de la competición M4 y almacena en
``data/raw/`` para evitar descargas repetidas en ejecuciones sucesivas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    CONFIG,
    M4_INFO_URL,
    M4_MONTHLY_TRAIN_URL,
    RAW_DIR,
    ExperimentConfig,
)
from src.utils import get_logger

_LOG = get_logger("data_loader")

_INFO_FILE = RAW_DIR / "M4-info.csv"
_TRAIN_FILE = RAW_DIR / "Monthly-train.csv"


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        _LOG.info("Usando caché local: %s", dest.name)
        return
    _LOG.info("Descargando %s ...", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv(dest, index=False)
    _LOG.info("Guardado en %s", dest)


def load_m4_finance(
    config: ExperimentConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el subconjunto Finance-Monthly del dataset M4.

    Descarga el CSV de metadatos y el CSV de entrenamiento si no están en
    caché; luego filtra las series de categoría Finance / Monthly.

    Args:
        config: Configuración del experimento (usa ``category`` y ``frequency``).

    Returns:
        Tupla ``(info_df, data_df)`` donde:
        - ``info_df``: DataFrame de metadatos indexado por M4id.
        - ``data_df``: DataFrame de series indexado por M4id, con columnas
          ``V1, V2, …`` representando los meses; valores ausentes = NaN.
    """
    _download_if_missing(M4_INFO_URL, _INFO_FILE)
    _download_if_missing(M4_MONTHLY_TRAIN_URL, _TRAIN_FILE)

    _LOG.info("Cargando metadatos ...")
    info = pd.read_csv(_INFO_FILE)
    mask = (info["category"] == config.category) & (info["SP"] == config.frequency)
    finance_info = info[mask].copy()

    _LOG.info("Series Finance-Monthly encontradas: %d", len(finance_info))

    _LOG.info("Cargando series de entrenamiento ...")
    train = pd.read_csv(_TRAIN_FILE)
    train.rename(columns={"V1": "M4id"}, inplace=True)
    finance_data = train[train["M4id"].isin(finance_info["M4id"])].set_index("M4id")

    finance_info = finance_info.set_index("M4id")

    _LOG.info("Dataset cargado: %d series × %d meses máximos", *finance_data.shape)
    return finance_info, finance_data

"""Exporta metadatos de las 200 series de la submuestra a un CSV.

Para cada serie exporta: identificador, cuartil de longitud (usado en la
estratificación), longitud real, número de folds walk-forward retenidos y
decil de volumen. Reutiliza exactamente las mismas funciones de src/preprocessor.py
y src/backtesting.py que run_experiment.py, con la misma SEED, para garantizar
consistencia con los resultados reportados en la memoria.

En particular, el decil de volumen se calcula sobre la ventana inicial de
entrenamiento (initial_train), no sobre la serie completa, replicando la
corrección aplicada en run_experiment.py: así la clasificación por decil no
incorpora información de ningún periodo de evaluación de ningún fold.

Uso:
    python export_series_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CONFIG, SEED, TABLES_DIR
from src.utils import set_global_seed, get_logger
from src.backtesting import generate_folds
from src.data_loader import load_m4_finance
from src.preprocessor import (
    filter_by_min_length,
    stratified_subsample,
    extract_series_list,
    get_volume_deciles,
)

set_global_seed(SEED)
logger = get_logger("export_series_metadata")

MAX_FOLDS = 5


def main():
    logger.info("Cargando datos M4 (misma config y semilla que run_experiment.py)...")
    info_df, data_df = load_m4_finance(CONFIG)
    info_filt, data_filt, lengths_filt = filter_by_min_length(info_df, data_df, CONFIG)
    info_sub, data_sub = stratified_subsample(info_filt, data_filt, lengths_filt, CONFIG, seed=SEED)
    series_list = extract_series_list(data_sub)

    quartiles_full = pd.qcut(lengths_filt, q=4, labels=False, duplicates="drop")
    n_folds_retenidos = [min(len(generate_folds(len(s), CONFIG)), MAX_FOLDS) for s in series_list]

    # Deciles de volumen sobre la ventana inicial de entrenamiento (no la serie
    # completa): evita incorporar información de ningún fold de evaluación.
    volume_deciles = get_volume_deciles(series_list, initial_train=CONFIG.initial_train)

    series_metadata = pd.DataFrame({
        "id": info_sub.index,
        "cuartil_longitud": quartiles_full.loc[info_sub.index].values,
        "longitud": info_sub["Actual_Length"].values,
        "n_folds": n_folds_retenidos,
        "decil_volumen": volume_deciles,
    })

    out_path = TABLES_DIR / "series_metadata.csv"
    series_metadata.to_csv(out_path, index=False)
    logger.info("Metadatos exportados: %s (%d series)", out_path, len(series_metadata))
    print(series_metadata.head())


if __name__ == "__main__":
    main()

"""Benchmark homogéneo de coste computacional para los siete modelos.

Mide, bajo el mismo protocolo y el mismo fold, el tiempo de ajuste y de
inferencia de los cuatro modelos locales (SNaive, ETS, Holt-Winters, SARIMA)
y de los tres modelos globales (LightGBM, MLP, N-BEATS), además del tamaño
serializado de cada modelo ajustado. Usa fold_id=0 (posición relativa más
antigua de la ventana retenida), disponible para las 200 series de la
submuestra, de modo que los siete modelos se evalúan sobre exactamente el
mismo conjunto de series.

Uso:
    python benchmark_timing.py
"""

from __future__ import annotations

import io
import pickle
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil

from src.config import CONFIG, SEED
from src.utils import set_global_seed, get_logger
from src.backtesting import generate_folds, split_series
from src.data_loader import load_m4_finance
from src.preprocessor import filter_by_min_length, stratified_subsample, extract_series_list

set_global_seed(SEED)
logger = get_logger("benchmark_timing")


def _model_size_bytes(obj) -> int:
    """Tamaño serializado aproximado de un modelo ajustado, en bytes."""
    try:
        import torch
        if hasattr(obj, "state_dict"):
            buf = io.BytesIO()
            torch.save(obj.state_dict(), buf)
            return buf.tell()
    except ImportError:
        pass
    try:
        return len(pickle.dumps(obj))
    except Exception:
        return -1


def benchmark_local(model_name, ModelClass, folds_by_series, series_list):
    fit_times, predict_times = [], []
    last_model = None
    for i, series in enumerate(series_list):
        fold = folds_by_series[i]
        if fold is None:
            continue
        train, _ = split_series(series, fold)
        m = ModelClass()
        t0 = time.perf_counter()
        m.fit(train)
        t1 = time.perf_counter()
        m.predict(CONFIG.horizon)
        t2 = time.perf_counter()
        fit_times.append(t1 - t0)
        predict_times.append(t2 - t1)
        last_model = m
    return {
        "modelo": model_name,
        "n_series_evaluadas": len(fit_times),
        "tiempo_ajuste_total_s": float(np.sum(fit_times)),
        "tiempo_ajuste_medio_s": float(np.mean(fit_times)),
        "tiempo_inferencia_total_s": float(np.sum(predict_times)),
        "tiempo_inferencia_media_s": float(np.mean(predict_times)),
        "tamano_serializado_bytes": _model_size_bytes(last_model),
    }


def benchmark_global(model_name, ModelClass, folds_by_series, series_list):
    trains = []
    for i, series in enumerate(series_list):
        fold = folds_by_series[i]
        if fold is None:
            continue
        train, _ = split_series(series, fold)
        trains.append(train)

    gm = ModelClass()
    t0 = time.perf_counter()
    gm.fit_global(trains)
    t1 = time.perf_counter()

    predict_times = []
    for train in trains:
        t2 = time.perf_counter()
        gm.predict(train, CONFIG.horizon)
        t3 = time.perf_counter()
        predict_times.append(t3 - t2)

    return {
        "modelo": model_name,
        "n_series_evaluadas": len(trains),
        "tiempo_ajuste_total_s": t1 - t0,
        "tiempo_ajuste_medio_s": (t1 - t0) / len(trains),
        "tiempo_inferencia_total_s": float(np.sum(predict_times)),
        "tiempo_inferencia_media_s": float(np.mean(predict_times)),
        "tamano_serializado_bytes": _model_size_bytes(gm._net if hasattr(gm, "_net") else gm._model),
    }


def main():
    logger.info("Cargando datos M4 (misma config y semilla que run_experiment.py)...")
    info_df, data_df = load_m4_finance(CONFIG)
    info_filt, data_filt, lengths_filt = filter_by_min_length(info_df, data_df, CONFIG)
    info_sub, data_sub = stratified_subsample(info_filt, data_filt, lengths_filt, CONFIG, seed=SEED)
    series_list = extract_series_list(data_sub)
    logger.info("Series cargadas: %d", len(series_list))

    # fold_id=0 (posición relativa más antigua de la ventana retenida) está
    # disponible para las 200 series, ya que toda serie de la submuestra
    # genera al menos 1 fold retenido.
    folds_by_series = []
    for s in series_list:
        all_folds = generate_folds(len(s), CONFIG)
        retained = all_folds[-5:]
        folds_by_series.append(retained[0] if retained else None)
    n_available = sum(f is not None for f in folds_by_series)
    logger.info("Series con fold_id=0 disponible: %d/%d", n_available, len(series_list))

    rows = []

    from src.models.baseline import SeasonalNaive
    logger.info("Benchmark: SNaive")
    rows.append(benchmark_local("SNaive", SeasonalNaive, folds_by_series, series_list))

    from src.models.statistical import ETSForecaster, HoltWintersForecaster, SARIMAForecaster
    for name, klass in [("ETS", ETSForecaster), ("HoltWinters", HoltWintersForecaster), ("SARIMA", SARIMAForecaster)]:
        logger.info("Benchmark: %s", name)
        rows.append(benchmark_local(name, klass, folds_by_series, series_list))

    from src.models.ml import LightGBMForecaster, MLPForecaster, NBEATSForecaster
    for name, klass in [("LightGBM", LightGBMForecaster), ("MLP", MLPForecaster), ("NBEATS", NBEATSForecaster)]:
        logger.info("Benchmark: %s", name)
        rows.append(benchmark_global(name, klass, folds_by_series, series_list))

    df = pd.DataFrame(rows)
    df["tamano_serializado_kb"] = (df["tamano_serializado_bytes"] / 1024).round(1)
    out_path = ROOT / "results" / "tables" / "resultados_timing.csv"
    df.to_csv(out_path, index=False)

    print("\n=== BENCHMARK DE COSTE COMPUTACIONAL (fold_id=0, 200 series, CPU) ===\n")
    print(df.drop(columns=["tamano_serializado_bytes"]).to_string(index=False))
    print(f"\nGuardado en: {out_path}")

    hw = {
        "sistema_operativo": f"{platform.system()} {platform.release()}",
        "procesador": "13th Gen Intel(R) Core(TM) i7-1355U",
        "cpu_fisicos": psutil.cpu_count(logical=False),
        "cpu_logicos": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python_version": platform.python_version(),
        "aceleracion_gpu": "No (CPU únicamente)",
    }
    hw_path = ROOT / "results" / "tables" / "entorno_hardware.csv"
    pd.DataFrame([hw]).to_csv(hw_path, index=False)
    print(f"\nEntorno de hardware guardado en: {hw_path}")
    for k, v in hw.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

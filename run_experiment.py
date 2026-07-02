"""Script standalone para ejecutar el experimento completo.

Alternativa al notebook para entornos donde Jupyter no está disponible.
Genera los mismos CSV y figuras que notebooks/main.ipynb.

Uso:
    python run_experiment.py [--n-series 200] [--no-dl] [--fast]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CONFIG, SEED, TABLES_DIR, FIGURES_DIR
from src.utils import set_global_seed, get_logger
from src.backtesting import generate_folds, split_series, verify_no_leakage
from src.evaluation import compute_all, bootstrap_smape_diff_ci
from src.data_loader import load_m4_finance
from src.preprocessor import (
    filter_by_min_length, stratified_subsample,
    extract_series_list, get_volume_deciles
)

set_global_seed(SEED)
logger = get_logger("run_experiment")

METRICS = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "WAPE"]
MODEL_ORDER = ["SNaive", "ETS", "HoltWinters", "SARIMA", "LightGBM", "MLP", "NBEATS"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-series", type=int, default=CONFIG.n_series or 200)
    p.add_argument("--max-folds", type=int, default=5,
                   help="Máximo de folds por serie (limita series muy largas)")
    p.add_argument("--no-dl", action="store_true", help="Omitir modelos de deep learning")
    p.add_argument("--fast", action="store_true", help="Modo rápido (2 series, 2 folds)")
    return p.parse_args()


def load_models(args):
    local, global_ = {}, {}

    from src.models.baseline import SeasonalNaive
    local["SNaive"] = SeasonalNaive
    logger.info("OK SNaive")

    for name, klass_path in [("ETS", "ETSForecaster"), ("HoltWinters", "HoltWintersForecaster"), ("SARIMA", "SARIMAForecaster")]:
        try:
            mod = __import__("src.models.statistical", fromlist=[klass_path])
            local[name] = getattr(mod, klass_path)
            logger.info("OK %s", name)
        except Exception as e:
            logger.warning("SKIP %s: %s", name, e)

    if not args.no_dl:
        for name, klass_path in [("LightGBM", "LightGBMForecaster"), ("MLP", "MLPForecaster"), ("NBEATS", "NBEATSForecaster")]:
            try:
                mod = __import__("src.models.ml", fromlist=[klass_path])
                global_[name] = getattr(mod, klass_path)
                logger.info("OK %s", name)
            except Exception as e:
                logger.warning("SKIP %s: %s", name, e)

    logger.info("Modelos locales: %s", list(local.keys()))
    logger.info("Modelos globales: %s", list(global_.keys()))
    return local, global_


def run_local(model_name, ModelClass, series_list, volume_deciles, config, max_folds=5):
    records = []
    for i, (series, decile) in enumerate(tqdm(
        zip(series_list, volume_deciles), total=len(series_list),
        desc=model_name, leave=False
    )):
        all_folds = generate_folds(len(series), config)
        # Usar los últimos max_folds folds (los más recientes, más relevantes)
        for fold in all_folds[-max_folds:]:
            train, test = split_series(series, fold)
            try:
                m = ModelClass()
                m.fit(train)
                pred = m.predict(config.horizon)
                mets = compute_all(test, pred, train, config.seasonality)
                rec = {"model": model_name, "series_idx": i, "fold_id": fold.fold_id, "decile": int(decile)}
                rec.update(mets)
            except Exception:
                rec = {"model": model_name, "series_idx": i, "fold_id": fold.fold_id,
                       "decile": int(decile), **{m: np.nan for m in METRICS}}
            records.append(rec)
    return records


def run_global(model_name, ModelClass, series_list, volume_deciles, config, fast=False, max_folds_global=5):
    records = []
    # Para globales: usar los últimos max_folds_global folds de cada serie
    all_series_folds = [generate_folds(len(s), config)[-max_folds_global:] for s in series_list]
    max_fold_idx = max(len(f) for f in all_series_folds)
    folds_to_run = range(min(2, max_fold_idx) if fast else max_fold_idx)

    for fold_id in tqdm(folds_to_run, desc=model_name, leave=False):
        trains, metas = [], []
        for i, (series, decile) in enumerate(zip(series_list, volume_deciles)):
            folds = all_series_folds[i]
            if fold_id < len(folds):
                fold = folds[fold_id]
                train, test = split_series(series, fold)
                trains.append(train)
                metas.append((i, fold, test, series, int(decile)))

        if not trains:
            continue

        try:
            gm = ModelClass()
            gm.fit_global(trains)
            for i, fold, test, series, decile in metas:
                train, _ = split_series(series, fold)
                try:
                    pred = gm.predict(train, config.horizon)
                    mets = compute_all(test, pred, train, config.seasonality)
                    rec = {"model": model_name, "series_idx": i, "fold_id": fold_id, "decile": decile}
                    rec.update(mets)
                except Exception:
                    rec = {"model": model_name, "series_idx": i, "fold_id": fold_id,
                           "decile": decile, **{m: np.nan for m in METRICS}}
                records.append(rec)
        except Exception as ex:
            logger.warning("Fold %d falló para %s: %s", fold_id, model_name, ex)

    return records


def main():
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    from dataclasses import replace
    cfg = replace(CONFIG, n_series=2 if args.fast else args.n_series)

    logger.info("Cargando datos M4...")
    info_df, data_df = load_m4_finance(cfg)
    info_filt, data_filt, lengths_filt = filter_by_min_length(info_df, data_df, cfg)
    info_sub, data_sub = stratified_subsample(info_filt, data_filt, lengths_filt, cfg, seed=SEED)
    series_list = extract_series_list(data_sub)
    volume_deciles = get_volume_deciles(series_list)

    logger.info("Series: %d | Walk-forward: initial=%d, horizon=%d, step=%d",
                len(series_list), cfg.initial_train, cfg.horizon, cfg.step)

    # Verificar leakage
    import random
    random.seed(SEED)
    for i in random.sample(range(len(series_list)), min(10, len(series_list))):
        verify_no_leakage(generate_folds(len(series_list[i]), cfg), len(series_list[i]))
    logger.info("Verificación leakage: OK")

    local_models, global_models = load_models(args)

    all_records = []

    max_folds = 2 if args.fast else args.max_folds
    logger.info("Max folds por serie: %d", max_folds)

    # Modelos locales
    for model_name, ModelClass in local_models.items():
        logger.info("Ejecutando local: %s", model_name)
        recs = run_local(model_name, ModelClass, series_list, volume_deciles, cfg, max_folds=max_folds)
        all_records.extend(recs)
        smape_mean = np.nanmean([r["sMAPE"] for r in recs if "sMAPE" in r])
        logger.info("  %s sMAPE: %.3f%%", model_name, smape_mean)

    # Modelos globales
    for model_name, ModelClass in global_models.items():
        logger.info("Ejecutando global: %s", model_name)
        recs = run_global(model_name, ModelClass, series_list, volume_deciles, cfg,
                          fast=args.fast, max_folds_global=max_folds)
        all_records.extend(recs)
        if recs:
            smape_mean = np.nanmean([r["sMAPE"] for r in recs if "sMAPE" in r])
            logger.info("  %s sMAPE: %.3f%%", model_name, smape_mean)

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(TABLES_DIR / "resultados_detalle.csv", index=False)

    # Tabla resumen
    present = [m for m in MODEL_ORDER if m in df_all["model"].unique()]
    numeric_rows = []
    for model in present:
        sub = df_all[df_all["model"] == model]
        row = {"model": model}
        for metric in METRICS:
            vals = sub[metric].dropna()
            row[metric + "_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[metric + "_std"] = vals.std() if len(vals) > 0 else np.nan
        numeric_rows.append(row)

    numeric_summary = pd.DataFrame(numeric_rows)
    numeric_summary.to_csv(TABLES_DIR / "resultados_comparativa.csv", index=False)

    print("\n=== TABLA COMPARATIVA (media ± desviación estándar) ===\n")
    for _, row in numeric_summary.iterrows():
        print(f"{row['model']:<15}", end="")
        for m in METRICS:
            print(f"  {m}={row[m+'_mean']:.3f}±{row[m+'_std']:.3f}", end="")
        print()

    # Criterios de éxito — referencia dinámica desde los resultados del experimento
    snv_row = numeric_summary[numeric_summary["model"] == "SNaive"]
    SNAVE_REF = float(snv_row["sMAPE_mean"].values[0]) if len(snv_row) > 0 else float("nan")
    print(f"\n=== CRITERIOS DE ÉXITO (referencia SNaive: {SNAVE_REF:.3f}%) ===\n")
    print(f"{'Modelo':<15} {'sMAPE':>9} {'Delta(pp)':>10} {'MASE':>8} {'CV%':>7} {'Supera?':>9}")
    print("-" * 60)
    for _, row in numeric_summary.iterrows():
        s_mean = row["sMAPE_mean"]
        s_std = row["sMAPE_std"]
        mase = row["MASE_mean"]
        diff = SNAVE_REF - s_mean
        cv = s_std / s_mean * 100 if s_mean > 0 else float("nan")
        supera = "SÍ" if diff >= 1.0 else ("marginal" if diff > 0 else "NO")
        print(f"{row['model']:<15} {s_mean:>9.3f} {diff:>+8.3f} {mase:>8.3f} {cv:>7.1f} {supera:>9}")

    # Bootstrap CI al 95 % — diferencia pareada por serie: sMAPE(modelo) − sMAPE(SNaive)
    # Unidad de remuestreo: serie (N=200). Método: percentil bootstrap, B=1000, SEED=42.
    logger.info("Calculando IC bootstrap (B=1000, alpha=0.05)...")

    def _series_smape_means(df: pd.DataFrame, model: str) -> pd.Series:
        """sMAPE medio por serie para un modelo dado."""
        sub = df[df["model"] == model][["series_idx", "sMAPE"]].dropna()
        return sub.groupby("series_idx")["sMAPE"].mean()

    snv_means = _series_smape_means(df_all, "SNaive")
    bootstrap_rows = []
    for model in present:
        if model == "SNaive":
            bootstrap_rows.append({
                "model": model,
                "diff_mean": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "significant": False,
                "n_series": len(snv_means),
            })
            continue
        model_means = _series_smape_means(df_all, model)
        common_idx = snv_means.index.intersection(model_means.index)
        if len(common_idx) < 2:
            logger.warning("  %s: series comunes insuficientes para bootstrap (%d)", model, len(common_idx))
            continue
        ci = bootstrap_smape_diff_ci(
            model_means[common_idx].values,
            snv_means[common_idx].values,
            B=1000,
            alpha=0.05,
            seed=SEED,
        )
        ci["model"] = model
        bootstrap_rows.append(ci)
        sign_str = "SIGNIFICATIVO" if ci["significant"] else "no significativo"
        logger.info(
            "  %s: diff=%.3f pp  IC95=[%.3f, %.3f]  %s",
            model, ci["diff_mean"], ci["ci_lower"], ci["ci_upper"], sign_str,
        )

    df_bootstrap = pd.DataFrame(bootstrap_rows)[
        ["model", "diff_mean", "ci_lower", "ci_upper", "significant", "n_series"]
    ]
    df_bootstrap.to_csv(TABLES_DIR / "resultados_bootstrap_ci.csv", index=False)
    logger.info("Bootstrap CI exportado a: %s", TABLES_DIR / "resultados_bootstrap_ci.csv")

    print("\n=== INTERVALOS DE CONFIANZA BOOTSTRAP al 95 % (modelo - SNaive, pp) ===\n")
    print(f"{'Modelo':<15} {'Diff(pp)':>10} {'IC_lower':>10} {'IC_upper':>10} {'Significativo':>14}")
    print("-" * 65)
    for _, row in df_bootstrap.iterrows():
        sig = "SÍ" if row["significant"] else "NO"
        print(f"{row['model']:<15} {row['diff_mean']:>+10.3f} {row['ci_lower']:>+10.3f} {row['ci_upper']:>+10.3f} {sig:>14}")

    # Generar figuras
    logger.info("Generando figuras...")
    try:
        from src.visualization import (
            plot_walk_forward_schema, plot_results_bar,
            plot_metrics_heatmap, plot_stability, plot_decile_contribution
        )

        plot_walk_forward_schema(filename="fig01_walk_forward_schema.png")

        plot_results_bar(df_all, metric="sMAPE", filename="fig02_smape_comparativa.png")

        hm_df = numeric_summary.rename(columns={m + "_mean": m for m in METRICS})
        plot_metrics_heatmap(hm_df, metrics=METRICS, filename="fig03_metricas_heatmap.png")

        df_local = df_all[df_all["model"].isin(local_models.keys())]
        if "fold_id" in df_local.columns:
            plot_stability(df_local, metric="sMAPE", filename="fig04_estabilidad_temporal.png")

        dec_rows = []
        for model in present:
            sub = df_all[df_all["model"] == model]
            for decile in sorted(sub["decile"].dropna().unique()):
                dsub = sub[sub["decile"] == decile]
                dec_rows.append({"model": model, "decile": int(decile),
                                 "abs_error": dsub["MAE"].dropna().sum()})
        dec_df = pd.DataFrame(dec_rows)
        total_err = dec_df.groupby("model")["abs_error"].transform("sum")
        dec_df["contribution"] = dec_df["abs_error"] / total_err * 100
        plot_decile_contribution(dec_df, filename="fig05_contribucion_deciles.png")

        logger.info("Figuras generadas en: %s", FIGURES_DIR)
    except Exception as e:
        logger.warning("Error generando figuras: %s", e)

    logger.info("Experimento completado. Resultados en: %s", TABLES_DIR)
    return df_all, numeric_summary


if __name__ == "__main__":
    df_all, summary = main()

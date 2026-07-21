"""Visualizaciones del experimento comparativo.

Genera figuras numeradas para la memoria: comparativa de modelos, diagrama
del esquema walk-forward, estabilidad temporal y contribución al error por
decil. Todas las figuras se guardan en ``results/figures/`` en formato PNG de
alta resolución.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

_PALETTE = sns.color_palette("tab10")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

_FIGDIR = Path(FIGURES_DIR)
_FIGDIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    dest = _FIGDIR / filename
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_walk_forward_schema(
    series_length: int = 120,
    initial_train: int = 54,
    horizon: int = 18,
    step: int = 12,
    n_folds: int = 4,
    filename: str = "fig01_walk_forward_schema.png",
) -> Path:
    """Diagrama del esquema de validación walk-forward expanding window."""
    fig, ax = plt.subplots(figsize=(10, 3.5))

    colors_train = "#2166ac"
    colors_test = "#d73027"
    colors_future = "#f7f7f7"

    train_end = initial_train
    for k in range(n_folds):
        test_end = train_end + horizon
        ax.barh(
            y=k, width=train_end, left=0, height=0.6, color=colors_train, alpha=0.85
        )
        ax.barh(
            y=k, width=horizon, left=train_end, height=0.6, color=colors_test, alpha=0.85
        )
        ax.text(
            train_end / 2, k, f"Train (fold {k})", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold"
        )
        ax.text(
            train_end + horizon / 2, k, f"Test {k}", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold"
        )
        train_end += step

    ax.set_xlabel("Meses")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {k}" for k in range(n_folds)])
    ax.set_xlim(0, series_length)
    ax.set_title("Esquema de validación walk-forward (expanding window)", fontweight="bold")
    legend = [
        mpatches.Patch(color=colors_train, label="Entrenamiento"),
        mpatches.Patch(color=colors_test, label="Evaluación (18 meses)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, filename)


def plot_results_bar(
    results_df: pd.DataFrame,
    metric: str = "sMAPE",
    title: Optional[str] = None,
    filename: str = "fig02_smape_comparativa.png",
) -> Path:
    """Gráfico de barras con el valor medio de una métrica por modelo."""
    order = results_df.groupby("model")[metric].mean().sort_values().index.tolist()
    means = results_df.groupby("model")[metric].mean().reindex(order)
    stds = results_df.groupby("model")[metric].std().reindex(order)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        range(len(order)),
        means.values,
        yerr=stds.values,
        capsize=4,
        color=_PALETTE[: len(order)],
        edgecolor="white",
        alpha=0.88,
    )
    for bar, val in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel(f"{metric} (%)" if "APE" in metric or "APE" in metric.upper() else metric)
    ax.set_title(title or f"Comparativa de modelos — {metric}", fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_metrics_heatmap(
    summary_df: pd.DataFrame,
    metrics: List[str],
    filename: str = "fig03_metricas_heatmap.png",
) -> Path:
    """Mapa de calor: modelos × métricas (valores normalizados por columna)."""
    pivot = summary_df.set_index("model")[metrics]
    normed = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.4, len(pivot) * 0.65 + 1))
    sns.heatmap(
        normed,
        ax=ax,
        annot=pivot.round(3),
        fmt=".3f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Valor normalizado [0–1]"},
    )
    ax.set_title("Tabla de métricas normalizada (menor = mejor)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, filename)


def plot_stability(
    results_df: pd.DataFrame,
    metric: str = "sMAPE",
    models: Optional[List[str]] = None,
    filename: str = "fig04_estabilidad_temporal.png",
    title: Optional[str] = None,
) -> Path:
    """Evolución del sMAPE promedio por fold walk-forward (solo modelos locales).

    Para que la comparación entre posiciones de fold sea válida, ``results_df``
    debe contener, para cada modelo, el mismo conjunto de series en todas las
    posiciones de fold (ver ``run_experiment.py``, donde se filtra a las series
    con historial completo antes de llamar a esta función). De lo contrario, la
    media de cada posición se calcula sobre subconjuntos de series distintos y
    la tendencia observada no es atribuible únicamente al efecto de la ventana
    de entrenamiento creciente.
    """
    if models is None:
        models = results_df["model"].unique().tolist()

    if "fold_id" not in results_df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "fold_id no disponible en resultados", ha="center", va="center")
        return _save(fig, filename)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(models):
        sub = results_df[results_df["model"] == m]
        fold_means = sub.groupby("fold_id")[metric].mean()
        ax.plot(fold_means.index, fold_means.values, marker="o", label=m,
                color=_PALETTE[i % len(_PALETTE)], linewidth=1.8)

    ax.set_xlabel("Fold walk-forward")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Estabilidad temporal — {metric} por fold", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, filename)


def plot_decile_contribution(
    contribution_df: pd.DataFrame,
    filename: str = "fig05_contribucion_deciles.png",
) -> Path:
    """Barras apiladas: contribución al error absoluto por decil y modelo."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    models = contribution_df["model"].unique().tolist()
    deciles = sorted(contribution_df["decile"].unique().tolist())

    bar_width = 0.8 / len(models)
    x = np.arange(len(deciles))

    for i, m in enumerate(models):
        sub = contribution_df[contribution_df["model"] == m].set_index("decile")
        vals = [sub.loc[d, "contribution"] if d in sub.index else 0 for d in deciles]
        ax.bar(
            x + i * bar_width,
            vals, width=bar_width, label=m,
            color=_PALETTE[i % len(_PALETTE)], alpha=0.85
        )

    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"D{d+1}" for d in deciles])
    ax.set_xlabel("Decil de volumen (D1=menor, D10=mayor)")
    ax.set_ylabel("Contribución al error absoluto total (%)")
    ax.set_title("Contribución al error por decil de volumen y modelo", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, filename)


def plot_example_forecast(
    history: np.ndarray,
    test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    series_id: str = "ejemplo",
    zoom: int = 48,
    filename: str = "fig06_ejemplo_pronostico.png",
) -> Path:
    """Visualiza el pronóstico de múltiples modelos sobre una serie de ejemplo."""
    fig, ax = plt.subplots(figsize=(12, 5))
    t_hist = np.arange(len(history))
    t_test = np.arange(len(history), len(history) + len(test))

    ax.plot(t_hist[-zoom:], history[-zoom:], color="black", linewidth=1.8,
            label="Historia (entrenamiento)")
    ax.plot(t_test, test, color="black", linestyle="--", marker="o",
            markersize=4, linewidth=1.5, label="Real (test)")

    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(t_test, pred, color=_PALETTE[i % len(_PALETTE)],
                linestyle="-", linewidth=1.5, marker="s", markersize=3, label=name)

    ax.axvline(len(history) - 1, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Meses")
    ax.set_ylabel("Valor")
    ax.set_title(f"Pronóstico comparativo — Serie {series_id}", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, filename)

"""Pruebas del motor walk-forward: corrección de los folds y ausencia de fuga.

Verifican que la generación de particiones respeta la causalidad temporal sobre
series sintéticas de distintas longitudes y bajo ambas variantes de ventana.
Ejecutable con ``pytest`` o directamente con ``python tests/test_backtesting.py``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting import Fold, generate_folds, verify_no_leakage  # noqa: E402
from src.config import CONFIG  # noqa: E402


def test_expanding_fold_count_and_geometry() -> None:
    # n=120, initial_train=54, horizon=18, step=12, expanding.
    # Cortes en train_end = 54, 66, ..., mientras train_end + 18 <= 120 => <=102.
    # 54,66,78,90,102 -> 5 folds.
    cfg = replace(CONFIG, initial_train=54, horizon=18, step=12, window="expanding")
    folds = generate_folds(120, cfg)
    assert len(folds) == 5
    # Todos los entrenamientos parten del origen en modo expanding.
    assert all(f.train_start == 0 for f in folds)
    assert folds[0].train_end == 54 and folds[0].test_end == 72
    assert folds[-1].train_end == 102 and folds[-1].test_end == 120


def test_rolling_keeps_fixed_window() -> None:
    cfg = replace(CONFIG, initial_train=54, horizon=18, step=12, window="rolling")
    folds = generate_folds(120, cfg)
    # En modo rolling el tamaño del entrenamiento es constante = initial_train.
    assert all((f.train_end - f.train_start) == 54 for f in folds)
    # Y la ventana se desplaza: el segundo fold empieza 12 meses después.
    assert folds[1].train_start == folds[0].train_start + 12


def test_no_leakage_multiple_lengths() -> None:
    cfg = replace(CONFIG, initial_train=54, horizon=18, step=12)
    for n in (72, 90, 120, 175, 240):
        for window in ("expanding", "rolling"):
            folds = generate_folds(n, replace(cfg, window=window))
            # No debe lanzar: confirma contigüidad, disyunción y límites.
            verify_no_leakage(folds, n)


def test_short_series_yields_no_fold() -> None:
    cfg = replace(CONFIG, initial_train=54, horizon=18, step=12)
    # n=71 < 54+18 => no cabe ninguna iteración.
    assert generate_folds(71, cfg) == []
    # n=72 => exactamente un fold (54 entrenamiento + 18 evaluación).
    assert len(generate_folds(72, cfg)) == 1


def test_test_window_is_always_future() -> None:
    cfg = replace(CONFIG, initial_train=54, horizon=18, step=12)
    folds = generate_folds(200, cfg)
    for f in folds:
        # Cada índice de evaluación es mayor que cualquier índice de entrenamiento.
        assert f.test_start >= f.train_end
        assert f.test_start == f.train_end  # contigüidad estricta


def test_detects_injected_leakage() -> None:
    # Un fold manipulado en el que la evaluación solapa el entrenamiento debe
    # ser detectado por verify_no_leakage.
    bad = Fold(fold_id=0, train_start=0, train_end=60, test_start=50, test_end=68)
    try:
        verify_no_leakage([bad], n=100)
    except AssertionError:
        return
    raise AssertionError("verify_no_leakage no detectó la fuga inyectada.")


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} pruebas superadas.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

"""Validación de las métricas frente a valores calculados manualmente.

Cada prueba contrasta la implementación con un resultado derivado a mano sobre
un ejemplo pequeño y conocido, de modo que un fallo señale inequívocamente una
desviación respecto de la definición formal de la métrica. Ejecutable tanto con
``pytest`` como directamente con ``python tests/test_evaluation.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402

TOL = 1e-6

# Ejemplo de referencia usado en varias pruebas.
Y_TRUE = [100.0, 200.0, 300.0]
Y_PRED = [110.0, 190.0, 330.0]
# Errores absolutos = [10, 10, 30]; suma = 50; suma|y_true| = 600.


def test_mae() -> None:
    # (10 + 10 + 30) / 3 = 16.666...
    assert math.isclose(ev.mae(Y_TRUE, Y_PRED), 50.0 / 3.0, rel_tol=TOL)


def test_rmse() -> None:
    # sqrt((100 + 100 + 900) / 3) = sqrt(366.666...) = 19.148542...
    assert math.isclose(ev.rmse(Y_TRUE, Y_PRED), math.sqrt(1100.0 / 3.0), rel_tol=TOL)


def test_mape() -> None:
    # mean(0.10, 0.05, 0.10) * 100 = 8.333...
    assert math.isclose(ev.mape(Y_TRUE, Y_PRED), 100.0 * (0.10 + 0.05 + 0.10) / 3.0,
                        rel_tol=TOL)


def test_smape() -> None:
    # mean(20/210, 20/390, 60/630) * 100
    expected = 100.0 * (20 / 210 + 20 / 390 + 60 / 630) / 3.0
    assert math.isclose(ev.smape(Y_TRUE, Y_PRED), expected, rel_tol=TOL)


def test_wape() -> None:
    # 50 / 600 * 100 = 8.333...
    assert math.isclose(ev.wape(Y_TRUE, Y_PRED), 50.0 / 600.0 * 100.0, rel_tol=TOL)


def test_mase_seasonality_one() -> None:
    # y_train = [1,2,3,4,5], m=1: errores naïve = [1,1,1,1], escala = 1.
    # MASE = MAE / escala = (50/3) / 1.
    result = ev.mase(Y_TRUE, Y_PRED, y_train=[1, 2, 3, 4, 5], seasonality=1)
    assert math.isclose(result, (50.0 / 3.0), rel_tol=TOL)


def test_mase_seasonality_two() -> None:
    # y_train = [1,3,2,5,4,7], m=2: diffs = [1,2,2,2], escala = 1.75.
    # MASE = (50/3) / 1.75 = 9.523809...
    result = ev.mase(Y_TRUE, Y_PRED, y_train=[1, 3, 2, 5, 4, 7], seasonality=2)
    assert math.isclose(result, (50.0 / 3.0) / 1.75, rel_tol=TOL)


def test_smape_handles_zero_pair() -> None:
    # Predicción perfecta de un cero no debe penalizar: término = 0.
    expected = 100.0 * (0.0 + 2 * 10 / 210) / 2.0
    assert math.isclose(ev.smape([0.0, 100.0], [0.0, 110.0]), expected, rel_tol=TOL)


def test_mape_excludes_zero_true() -> None:
    # El término con y_true = 0 se excluye; queda mean(10/100) * 100 = 10.
    assert math.isclose(ev.mape([0.0, 100.0], [5.0, 110.0]), 10.0, rel_tol=TOL)


def test_error_contribution() -> None:
    result = ev.error_contribution({"a": 30.0, "b": 10.0})
    assert math.isclose(result["a"], 0.75, rel_tol=TOL)
    assert math.isclose(result["b"], 0.25, rel_tol=TOL)
    assert math.isclose(sum(result.values()), 1.0, rel_tol=TOL)


def test_perfect_forecast_is_zero() -> None:
    assert ev.mae(Y_TRUE, Y_TRUE) == 0.0
    assert ev.rmse(Y_TRUE, Y_TRUE) == 0.0
    assert ev.smape(Y_TRUE, Y_TRUE) == 0.0
    assert ev.wape(Y_TRUE, Y_TRUE) == 0.0


def test_length_mismatch_raises() -> None:
    try:
        ev.mae([1.0, 2.0], [1.0])
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError por longitudes distintas.")


def test_nonfinite_raises() -> None:
    try:
        ev.mae([1.0, float("nan")], [1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError por valores no finitos.")


def test_empty_raises() -> None:
    try:
        ev.mae([], [])
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError por entrada vacía.")


def _run_all() -> int:
    """Ejecuta todas las pruebas sin depender de pytest. Devuelve nº de fallos."""
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

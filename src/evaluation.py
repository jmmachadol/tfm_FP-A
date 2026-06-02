"""Métricas de evaluación de pronósticos.

Implementa las métricas estadísticas (MAE, RMSE, MAPE, sMAPE, MASE) y de
impacto económico (WAPE y contribución al error absoluto) empleadas en la
comparativa. Cada función incluye su definición formal, de modo que la
implementación sea verificable frente a la literatura de referencia
(Hyndman & Athanasopoulos, 2021; Petropoulos et al., 2022).

Convención de argumentos:
    y_true: valores reales observados en el horizonte de evaluación.
    y_pred: valores pronosticados para el mismo horizonte.
Ambos se aceptan como cualquier estructura convertible a ``np.ndarray`` de una
dimensión y deben tener la misma longitud.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | list[float] | pd.Series

_EPS = 1e-8  # Constante de estabilidad para evitar divisiones por cero.


def _as_1d_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convierte las entradas a arrays 1-D de float y valida su consistencia.

    Args:
        y_true: Valores reales.
        y_pred: Valores pronosticados.

    Returns:
        Tupla ``(y_true, y_pred)`` como arrays 1-D de tipo float.

    Raises:
        ValueError: Si las longitudes difieren, si están vacíos o si contienen
            valores no finitos (NaN o infinito).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if yt.size == 0 or yp.size == 0:
        raise ValueError("Las series de evaluación no pueden estar vacías.")
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true y y_pred deben tener la misma longitud; "
            f"se recibió {yt.shape} y {yp.shape}."
        )
    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("y_true e y_pred no pueden contener NaN ni infinitos.")
    return yt, yp


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Error absoluto medio. :math:`\mathrm{MAE}=\frac1n\sum|y_t-\hat y_t|`."""
    yt, yp = _as_1d_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Raíz del error cuadrático medio.
    :math:`\mathrm{RMSE}=\sqrt{\frac1n\sum(y_t-\hat y_t)^2}`."""
    yt, yp = _as_1d_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Error porcentual absoluto medio, en porcentaje.
    :math:`\mathrm{MAPE}=\frac{100}{n}\sum\frac{|y_t-\hat y_t|}{|y_t|}`.

    Los términos con :math:`y_t=0` se excluyen del promedio, dado que el MAPE no
    está definido en ese punto. Si todos los valores reales son cero, devuelve
    ``nan``.
    """
    yt, yp = _as_1d_arrays(y_true, y_pred)
    mask = np.abs(yt) > _EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / np.abs(yt[mask])) * 100.0)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""sMAPE simétrico en la forma de la competición M4, en porcentaje.

    :math:`\mathrm{sMAPE}=\frac{100}{n}\sum\frac{2|y_t-\hat y_t|}{|y_t|+|\hat y_t|}`,
    acotado en [0, 200]. Los términos en los que numerador y denominador son
    ambos cero (predicción perfecta de un valor nulo) contribuyen 0.
    """
    yt, yp = _as_1d_arrays(y_true, y_pred)
    numerator = np.abs(yt - yp)
    denominator = (np.abs(yt) + np.abs(yp)) / 2.0
    ratio = np.where(denominator < _EPS, 0.0, numerator / np.maximum(denominator, _EPS))
    return float(np.mean(ratio) * 100.0)


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 12,
) -> float:
    r"""Error absoluto escalado medio (Mean Absolute Scaled Error).

    Escala el MAE del pronóstico por el MAE en muestra de un pronóstico naïve
    estacional sobre los datos de entrenamiento:

    .. math::
        \mathrm{MASE}=\frac{\frac1h\sum_{t}|y_t-\hat y_t|}
        {\frac{1}{n-m}\sum_{j=m+1}^{n}|y_j-y_{j-m}|}

    donde :math:`m` es la estacionalidad y la suma del denominador recorre la
    serie de entrenamiento. Un valor < 1 indica que el modelo supera al naïve
    estacional. Es la definición de Hyndman & Koehler (2006).

    Args:
        y_true: Valores reales del horizonte.
        y_pred: Valores pronosticados.
        y_train: Serie de entrenamiento sobre la que se calcula la escala.
        seasonality: Periodo estacional ``m`` del naïve de referencia.

    Returns:
        El MASE. Devuelve ``nan`` si la escala es nula (serie de entrenamiento
        perfectamente periódica con desviación cero).

    Raises:
        ValueError: Si la serie de entrenamiento es demasiado corta para el
            periodo estacional indicado o ``seasonality`` no es positivo.
    """
    yt, yp = _as_1d_arrays(y_true, y_pred)
    train = np.asarray(y_train, dtype=float).ravel()
    if seasonality < 1:
        raise ValueError("seasonality debe ser un entero positivo.")
    if train.size <= seasonality:
        raise ValueError(
            f"La serie de entrenamiento (n={train.size}) debe superar el "
            f"periodo estacional (m={seasonality}) para calcular el MASE."
        )

    naive_errors = np.abs(train[seasonality:] - train[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale < _EPS:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Error porcentual absoluto ponderado, en porcentaje.

    :math:`\mathrm{WAPE}=\frac{\sum|y_t-\hat y_t|}{\sum|y_t|}\times 100`.

    Pondera implícitamente cada error por la magnitud del valor real, de modo
    que los periodos (o series, al agregarse) de mayor volumen financiero pesan
    más. Devuelve ``nan`` si el volumen total es nulo.
    """
    yt, yp = _as_1d_arrays(y_true, y_pred)
    total = float(np.sum(np.abs(yt)))
    if total < _EPS:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / total * 100.0)


def error_contribution(
    abs_errors_by_group: Mapping[str, float],
) -> dict[str, float]:
    """Contribución relativa de cada grupo al error absoluto total.

    Dado el error absoluto acumulado por grupo (por ejemplo, por decil de
    volumen de serie), calcula qué fracción del error total aporta cada grupo.
    Permite identificar dónde se concentra el error y, por tanto, dónde una
    mejora del modelo tiene mayor impacto económico.

    Args:
        abs_errors_by_group: Mapa ``grupo -> error absoluto acumulado``.

    Returns:
        Mapa ``grupo -> proporción (0-1)`` cuyas proporciones suman 1. Si el
        error total es nulo, devuelve 0 para todos los grupos.
    """
    total = float(sum(abs_errors_by_group.values()))
    if total < _EPS:
        return {g: 0.0 for g in abs_errors_by_group}
    return {g: float(v) / total for g, v in abs_errors_by_group.items()}


def compute_all(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 12,
) -> dict[str, float]:
    """Calcula el conjunto completo de métricas estadísticas para una serie.

    Args:
        y_true: Valores reales del horizonte.
        y_pred: Valores pronosticados.
        y_train: Serie de entrenamiento (necesaria para el MASE).
        seasonality: Periodo estacional.

    Returns:
        Diccionario con las métricas MAE, RMSE, MAPE, sMAPE, MASE y WAPE.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality),
        "WAPE": wape(y_true, y_pred),
    }

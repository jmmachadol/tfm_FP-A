"""Motor de validación temporal walk-forward (backtesting con ventanas móviles).

La evaluación de modelos sobre series temporales no puede emplear una partición
aleatoria ni un único hold-out: la primera rompe la causalidad temporal y la
segunda no captura la variabilidad del desempeño en el tiempo. Bergmeir y
Hyndman (2012) demostraron que ambas producen estimaciones de error
artificialmente optimistas. El protocolo walk-forward resuelve este problema
generando, para cada serie, múltiples pares (entrenamiento, evaluación)
desplazados en el tiempo, en los que el conjunto de evaluación siempre es
posterior al de entrenamiento.

Este módulo genera exclusivamente los índices de cada partición; no ajusta
modelos. Esa separación permite verificar de forma aislada la ausencia de fuga
de información (ver ``verify_no_leakage`` y las pruebas asociadas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import CONFIG, ExperimentConfig


@dataclass(frozen=True)
class Fold:
    """Una iteración del walk-forward, expresada como índices semiabiertos.

    Los índices siguen la convención de Python ``[inicio, fin)``. El conjunto de
    entrenamiento es ``serie[train_start:train_end]`` y el de evaluación es
    ``serie[test_start:test_end]``, con ``test_start == train_end`` para que no
    exista solapamiento ni hueco entre ambos.

    Attributes:
        fold_id: Índice de la iteración (0 en la primera).
        train_start: Primer índice del entrenamiento (inclusive).
        train_end: Índice final del entrenamiento (exclusivo).
        test_start: Primer índice de la evaluación (inclusive).
        test_end: Índice final de la evaluación (exclusivo).
    """

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        """Slice del conjunto de entrenamiento."""
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        """Slice del conjunto de evaluación."""
        return slice(self.test_start, self.test_end)


def generate_folds(n: int, config: ExperimentConfig = CONFIG) -> list[Fold]:
    """Genera los folds walk-forward para una serie de longitud ``n``.

    El primer corte se sitúa en ``initial_train``; cada iteración avanza ``step``
    meses. La generación se detiene cuando ya no caben ``horizon`` observaciones
    de evaluación. En modo ``expanding`` el entrenamiento parte siempre del
    origen; en modo ``rolling`` mantiene un tamaño fijo de ``initial_train``.

    Args:
        n: Longitud de la serie en número de observaciones.
        config: Configuración del experimento (define initial_train, horizon,
            step y tipo de ventana).

    Returns:
        Lista de ``Fold`` ordenados temporalmente. Puede estar vacía si la serie
        es demasiado corta para generar al menos una iteración.

    Raises:
        ValueError: Si los parámetros del protocolo no son positivos o el tipo
            de ventana no se reconoce.
    """
    if config.initial_train <= 0 or config.horizon <= 0 or config.step <= 0:
        raise ValueError("initial_train, horizon y step deben ser positivos.")
    if config.window not in ("expanding", "rolling"):
        raise ValueError(f"Tipo de ventana no reconocido: {config.window!r}.")

    folds: list[Fold] = []
    fold_id = 0
    train_end = config.initial_train
    while train_end + config.horizon <= n:
        train_start = 0 if config.window == "expanding" else train_end - config.initial_train
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=max(0, train_start),
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + config.horizon,
            )
        )
        fold_id += 1
        train_end += config.step
    return folds


def verify_no_leakage(folds: list[Fold], n: int) -> None:
    """Verifica que un conjunto de folds no introduce fuga de información.

    Comprueba, para cada fold, las cinco condiciones que garantizan la validez
    temporal del protocolo:

    1. El entrenamiento es no vacío y la evaluación tiene exactamente el
       horizonte previsto.
    2. La evaluación comienza justo donde termina el entrenamiento
       (sin solapamiento ni hueco).
    3. Los índices de entrenamiento y evaluación son disjuntos.
    4. Todo el fold cae dentro de los límites de la serie ``[0, n)``.
    5. Cada evaluación es estrictamente posterior a la anterior (avance
       monótono en el tiempo).

    Args:
        folds: Folds a verificar.
        n: Longitud de la serie.

    Raises:
        AssertionError: Si alguna condición de no-fuga se incumple.
    """
    prev_test_start = -1
    for f in folds:
        assert f.train_end > f.train_start, f"Fold {f.fold_id}: entrenamiento vacío."
        assert f.test_end - f.test_start > 0, f"Fold {f.fold_id}: evaluación vacía."
        assert f.test_start == f.train_end, (
            f"Fold {f.fold_id}: la evaluación no es contigua al entrenamiento "
            f"(train_end={f.train_end}, test_start={f.test_start})."
        )
        train_idx = set(range(f.train_start, f.train_end))
        test_idx = set(range(f.test_start, f.test_end))
        assert train_idx.isdisjoint(test_idx), (
            f"Fold {f.fold_id}: solapamiento entre entrenamiento y evaluación."
        )
        assert 0 <= f.train_start < f.train_end <= f.test_start < f.test_end <= n, (
            f"Fold {f.fold_id}: índices fuera de los límites [0, {n})."
        )
        assert f.test_start > prev_test_start, (
            f"Fold {f.fold_id}: la evaluación no avanza en el tiempo."
        )
        prev_test_start = f.test_start


def split_series(series: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve las porciones de entrenamiento y evaluación de un fold.

    Args:
        series: Serie temporal completa (1-D).
        fold: Fold cuyas porciones se extraen.

    Returns:
        Tupla ``(train, test)`` con las dos porciones de la serie.
    """
    arr = np.asarray(series, dtype=float).ravel()
    return arr[fold.train_slice], arr[fold.test_slice]

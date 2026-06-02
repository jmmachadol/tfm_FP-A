"""Utilidades transversales: reproducibilidad y registro de eventos."""

from __future__ import annotations

import logging
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fija las semillas de todas las fuentes de aleatoriedad del pipeline.

    Cubre el módulo ``random`` de la biblioteca estándar, NumPy, la variable de
    entorno ``PYTHONHASHSEED`` y, si PyTorch está disponible, sus generadores de
    CPU y GPU junto con el modo determinista de cuDNN. Llamar a esta función al
    inicio de cada ejecución garantiza la reproducibilidad de los resultados.

    Args:
        seed: Valor entero no negativo para inicializar los generadores.

    Raises:
        ValueError: Si ``seed`` es negativo.
    """
    if seed < 0:
        raise ValueError(f"La semilla debe ser no negativa; se recibió {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch no está instalado: los modelos de deep learning no se usarán.
        pass


def get_logger(name: str = "tfm") -> logging.Logger:
    """Devuelve un logger configurado con formato homogéneo.

    Args:
        name: Nombre del logger.

    Returns:
        Un ``logging.Logger`` que escribe en consola con marca temporal y nivel.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

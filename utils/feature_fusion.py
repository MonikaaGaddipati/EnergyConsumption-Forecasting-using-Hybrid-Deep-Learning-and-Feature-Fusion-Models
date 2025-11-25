
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def simple_concatenate(historical_seq: np.ndarray, exogenous_seq: np.ndarray) -> np.ndarray:

    try:
        return np.concatenate([historical_seq, exogenous_seq], axis=-1)
    except Exception:
        logger.exception("Failed to concatenate sequences for fusion.")
        raise

"""Random seed utilities."""

import os
import random
import numpy as np
import tensorflow as tf


def set_all_seeds(seed):
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

"""Inference utilities for generating synthetic time series."""

import os
import numpy as np
import tensorflow as tf
from quantgan.models import build_generator
from quantgan.utils.io import assert_weights_compatible


def build_and_load_generator(model_cfg, window_len, weights_path, seed=0):
    netG = build_generator(model_cfg)

    burn_in = int(window_len - 1)
    _z = tf.random.normal([1, 10 + burn_in, int(model_cfg.z_dim)], seed=int(seed))
    _ = netG(_z, training=False)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    assert_weights_compatible(weights_path, model_cfg)

    import pickle
    with open(weights_path, "rb") as f:
        netG.set_weights(pickle.load(f))

    return netG


def generate_M_paths_raw(
    netG,
    preproc,
    M,
    Ttilde,
    window_len,
    z_dim,
    batch=50,
    seed=0
):
    """Generate M paths of length Ttilde in raw (original) scale.
    
    This function:
    1. Generates latent noise z
    2. Passes through generator to get processed returns
    3. Inverse-transforms back to original scale using preprocessor
    
    Args:
        netG: Generator model
        preproc: Preprocessor with inverse_transform method
        M: Number of paths to generate
        Ttilde: Length of each path
        window_len: Window length (for burn-in)
        z_dim: Latent dimension
        batch: Batch size for generation
        seed: Random seed
        
    Returns:
        Array of shape (M, Ttilde) with raw log-returns
    """
    burn_in = int(window_len - 1)
    M = int(M)
    Ttilde = int(Ttilde)
    z_dim = int(z_dim)
    batch = int(batch)

    g = tf.random.Generator.from_seed(int(seed))

    outs = np.zeros((M, Ttilde), dtype=np.float64)
    done = 0
    while done < M:
        b = min(batch, M - done)
        z = g.normal([b, Ttilde + burn_in, z_dim])
        fake_full = netG(z, training=False).numpy()[..., 0]  # (b, Ttilde+burn_in)
        fake_used = fake_full[:, burn_in:burn_in + Ttilde]
        outs[done:done + b] = preproc.inverse_transform(fake_used)
        done += b

    return outs

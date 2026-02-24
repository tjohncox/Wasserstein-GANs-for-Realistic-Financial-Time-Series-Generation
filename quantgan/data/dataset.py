"""Dataset creation utilities."""

import numpy as np
import tensorflow as tf


def make_windows_np(x, win):
    """Create sliding windows from 1D array.
    
    Args:
        x: 1D array
        win: Window size
        
    Returns:
        2D array of shape (N, win)
    """
    x = np.asarray(x, dtype=np.float32)
    win = int(win)
    if len(x) < win:
        raise ValueError(f"Not enough data for windows: len(x)={len(x)} win={win}")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        X = sliding_window_view(x, window_shape=win)
        return np.asarray(X, dtype=np.float32)
    except Exception:
        N = len(x) - win + 1
        return np.stack([x[i:i + win] for i in range(N)], axis=0).astype(np.float32)


def window_sampling_probs(series_len, win_len):
    """Compute sampling probabilities for windows.
    
    This ensures that each timestep appears in approximately the same
    number of windows during training.
    
    Args:
        series_len: Length of the time series
        win_len: Window length
        
    Returns:
        Tuple of (probabilities, counts)
    """
    series_len = int(series_len)
    win_len = int(win_len)
    Nw = series_len - win_len + 1

    count = np.zeros(series_len, dtype=np.int64)
    for t in range(series_len):
        s_min = max(0, t - win_len + 1)
        s_max = min(t, Nw - 1)
        count[t] = max(0, s_max - s_min + 1)

    inv = 1.0 / np.maximum(count, 1)
    w = np.zeros(Nw, dtype=np.float64)
    for s in range(Nw):
        w[s] = float(inv[s:s + win_len].sum())

    w = np.maximum(w, 1e-12)
    p = w / w.sum()
    return p, count


class DatasetBuilder:
    """Dataset builder for windowed time series."""

    def __init__(self, cfg):
        """Initialize dataset builder.
        
        Args:
            cfg: DatasetConfig instance
        """
        self.cfg = cfg

    def build(self, r_train):
        """Build TensorFlow dataset from preprocessed returns.
        
        Args:
            r_train: Preprocessed returns (1D array)
            
        Returns:
            Tuple of (tf.data.Dataset, windows_array, steps_per_epoch)
        """
        win = int(self.cfg.window_len)
        B = int(self.cfg.batch_size)

        X = make_windows_np(r_train, win)[:, :, None]  # (N, win, 1)
        Nw = int(X.shape[0])
        steps_per_epoch = max(1, Nw // B)

        if self.cfg.weighted_sampling:
            P_WIN, _ = window_sampling_probs(len(r_train), win)
            if B > Nw:
                raise ValueError(
                    f"BATCH({B}) > Nw({Nw}). Reduce batch_size or window_len."
                )

            rng = np.random.default_rng(self.cfg.seed)

            def gen():
                while True:
                    idx = rng.choice(Nw, size=B, replace=False, p=P_WIN)
                    yield X[idx]

            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=tf.TensorSpec(shape=(B, win, 1), dtype=tf.float32)
            ).prefetch(tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices(X)
            ds = ds.shuffle(Nw, seed=self.cfg.seed, reshuffle_each_iteration=True)
            ds = ds.batch(B, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        return ds, X, steps_per_epoch

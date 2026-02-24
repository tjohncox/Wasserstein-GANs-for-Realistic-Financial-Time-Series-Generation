"""Metrics for evaluating generated time series."""

import numpy as np
import tensorflow as tf


# ---------- TensorFlow metrics ----------

def acf_tf(x, lags):
    """Compute autocorrelation function in TensorFlow.
    
    Args:
        x: Tensor of shape (B, T, 1)
        lags: Number of lags
        
    Returns:
        Tensor of shape (B, lags)
    """
    x = x[..., 0]
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)

    outs = []
    for k in range(1, lags + 1):
        x_lagged = x[:, :-k]
        x_future = x[:, k:]
        num = tf.reduce_sum(x_lagged * x_future, axis=1)
        den = tf.sqrt(
            (tf.reduce_sum(x_lagged * x_lagged, axis=1) + 1e-12) *
            (tf.reduce_sum(x_future * x_future, axis=1) + 1e-12)
        )
        outs.append(num / den)

    return tf.stack(outs, axis=1)  # (B, lags)


def leverage_tf(r, nlags=40):
    """Compute leverage correlation in TensorFlow.
    
    Leverage correlation: corr(r_t, r^2_{t+k})
    
    Args:
        r: Returns tensor of shape (B, T, 1)
        nlags: Number of lags
        
    Returns:
        Tensor of shape (B, nlags)
    """
    x = r[..., 0]
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)
    y = tf.square(r[..., 0])
    y = y - tf.reduce_mean(y, axis=1, keepdims=True)

    out = []
    for k in range(1, nlags + 1):
        x_lagged = x[:, :-k]
        y_future = y[:, k:]
        cov = tf.reduce_sum(x_lagged * y_future, axis=1, keepdims=True)
        x_var = tf.reduce_sum(tf.square(x_lagged), axis=1, keepdims=True) + 1e-12
        y_var = tf.reduce_sum(tf.square(y_future), axis=1, keepdims=True) + 1e-12
        corr = cov / tf.sqrt(x_var * y_var)
        out.append(corr)
    return tf.concat(out, axis=1)


def tf_kurtosis_per_batch(x):
    """Compute kurtosis per batch in TensorFlow.
    
    Args:
        x: Tensor of shape (B, T, 1)
        
    Returns:
        Scalar kurtosis (averaged over batches)
    """
    y = x[..., 0]
    m = tf.reduce_mean(y, axis=1, keepdims=True)
    v = tf.reduce_mean(tf.square(y - m), axis=1) + 1e-12
    c4 = tf.reduce_mean(tf.pow(y - m, 4), axis=1)
    k = c4 / (v * v)
    return tf.reduce_mean(k)


# ---------- NumPy metrics ----------

def agg_returns_overlapping(r, t):
    """Aggregate returns over t periods using overlapping windows.
    
    Args:
        r: 1D array of returns
        t: Aggregation period
        
    Returns:
        Aggregated returns
    """
    r = np.asarray(r, dtype=np.float64)
    t = int(t)
    if t == 1:
        return r.copy()
    if len(r) < t:
        return np.array([], dtype=np.float64)
    return np.convolve(r, np.ones(t, dtype=np.float64), mode="valid")


def dy_metric(
    real_t,
    fake_t,
    dy_base_t,
    mass_per_bin=5,
    alpha=1e-8,
    merge_empty=False,
    max_bins=None,
    return_bins=False,
):
    """Compute DY (Density-Yield) metric.
    
    This metric measures the difference in distribution between real and fake
    aggregated returns.
    
    Args:
        real_t: Real aggregated returns
        fake_t: Fake aggregated returns
        dy_base_t: Base period for binning
        mass_per_bin: Target mass per bin
        alpha: Laplace smoothing parameter
        merge_empty: Merge empty bins
        max_bins: Maximum number of bins
        return_bins: Return number of bins used
        
    Returns:
        DY metric value (and number of bins if return_bins=True)
    """
    real_t = np.asarray(real_t, dtype=np.float64).reshape(-1)
    fake_t = np.asarray(fake_t, dtype=np.float64).reshape(-1)
    dy_base_t = int(dy_base_t)

    if len(real_t) < 10 or len(fake_t) < 10:
        return (np.nan, 0) if return_bins else np.nan

    n_bins = int(max(20, np.floor(dy_base_t / float(mass_per_bin))))
    if max_bins is not None:
        n_bins = int(min(n_bins, int(max_bins)))

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(real_t, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return (np.nan, 0) if return_bins else np.nan

    if merge_empty:
        for _ in range(1000):
            h_cnt, _ = np.histogram(real_t, bins=edges)
            g_cnt, _ = np.histogram(fake_t, bins=edges)
            empty = np.where(g_cnt == 0)[0]
            if len(empty) == 0 or len(edges) <= 3:
                break
            i = int(empty[0])
            if i == 0:
                del_idx = 1
            elif i >= len(edges) - 2:
                del_idx = len(edges) - 2
            else:
                del_idx = i + 1
            edges = np.delete(edges, del_idx)

    h_cnt, _ = np.histogram(real_t, bins=edges)
    g_cnt, _ = np.histogram(fake_t, bins=edges)

    widths = np.diff(edges).astype(np.float64)
    widths = np.maximum(widths, 1e-10)

    nb = len(h_cnt)
    ph = (h_cnt.astype(np.float64) + alpha) / (np.sum(h_cnt) + alpha * nb)
    pg = (g_cnt.astype(np.float64) + alpha) / (np.sum(g_cnt) + alpha * nb)

    ph_dens = ph / widths
    pg_dens = pg / widths

    dy = float(np.sum(np.abs(np.log(ph_dens + 1e-300) - np.log(pg_dens + 1e-300))))
    return (dy, nb) if return_bins else dy


def acf_vec(x, max_lags):
    """Compute autocorrelation function for a single series.
    
    Args:
        x: 1D array
        max_lags: Number of lags
        
    Returns:
        ACF values of shape (max_lags,)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    max_lags = int(max_lags)
    
    # Optimize: compute only up to min(max_lags, len(x)-1)
    effective_lags = min(max_lags, len(x) - 1)
    out = np.zeros(max_lags, dtype=np.float64)
    
    for k in range(1, effective_lags + 1):
        x_lagged = x[:-k]
        x_future = x[k:]
        num = np.dot(x_lagged, x_future)
        den = np.sqrt((np.dot(x_lagged, x_lagged) + 1e-12) * (np.dot(x_future, x_future) + 1e-12))
        out[k - 1] = num / den
    
    return out


def lev_vec(r, max_lags):
    """Compute leverage correlation for a single series.
    
    Args:
        r: 1D array of returns
        max_lags: Number of lags
        
    Returns:
        Leverage correlation values of shape (max_lags,)
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    max_lags = int(max_lags)
    x = r - r.mean()
    y = (r * r) - (r * r).mean()
    
    # Optimize: compute only up to min(max_lags, len(r)-1)
    effective_lags = min(max_lags, len(r) - 1)
    out = np.zeros(max_lags, dtype=np.float64)
    
    for k in range(1, effective_lags + 1):
        x_lagged = x[:-k]
        y_future = y[k:]
        num = np.dot(x_lagged, y_future)
        den = np.sqrt((np.dot(x_lagged, x_lagged) + 1e-12) * (np.dot(y_future, y_future) + 1e-12))
        out[k - 1] = num / den
    
    return out


def paper_dependence_scores(real_r, fake_paths, max_lags=250):
    """Compute dependence scores (ACF and leverage).
    
    Args:
        real_r: Real returns (1D)
        fake_paths: Fake returns (2D: M x T)
        max_lags: Number of lags
        
    Returns:
        Dictionary with scores
    """
    real_r = np.asarray(real_r, dtype=np.float64)
    fake_paths = np.asarray(fake_paths, dtype=np.float64)
    max_lags = int(max_lags)

    C_real_x = acf_vec(real_r, max_lags)
    C_real_abs = acf_vec(np.abs(real_r), max_lags)
    C_real_sq = acf_vec(real_r * real_r, max_lags)
    L_real = lev_vec(real_r, max_lags)

    C_fake_x = np.mean([acf_vec(fake_paths[i], max_lags) for i in range(fake_paths.shape[0])], axis=0)
    C_fake_abs = np.mean([acf_vec(np.abs(fake_paths[i]), max_lags) for i in range(fake_paths.shape[0])], axis=0)
    C_fake_sq = np.mean([acf_vec(fake_paths[i] * fake_paths[i], max_lags) for i in range(fake_paths.shape[0])], axis=0)
    L_fake = np.mean([lev_vec(fake_paths[i], max_lags) for i in range(fake_paths.shape[0])], axis=0)

    scores = {
        "acf_x": float(np.sqrt(np.mean((C_real_x - C_fake_x) ** 2))),
        "acf_abs": float(np.sqrt(np.mean((C_real_abs - C_fake_abs) ** 2))),
        "acf_sq": float(np.sqrt(np.mean((C_real_sq - C_fake_sq) ** 2))),
        "lev": float(np.sqrt(np.sum((L_real - L_fake) ** 2))),
    }
    return scores


def paper_distribution_metrics(real_r, fake_paths, dy_base_t, t_lags=(1, 5, 20, 100)):
    """Compute distribution metrics (DY) for multiple aggregation periods.
    
    Args:
        real_r: Real returns (1D)
        fake_paths: Fake returns (2D: M x T)
        dy_base_t: Base period for DY metric
        t_lags: Aggregation periods
        
    Returns:
        Dictionary with DY metrics for each aggregation period
    """
    real_r = np.asarray(real_r, dtype=np.float64)
    fake_paths = np.asarray(fake_paths, dtype=np.float64)
    dy_base_t = int(dy_base_t)

    if fake_paths.ndim == 1:
        fake_paths = fake_paths.reshape(1, -1)

    def agg_over_paths(paths2d, t):
        paths2d = np.asarray(paths2d, dtype=np.float64)
        t = int(t)
        if t == 1:
            return paths2d.reshape(-1)
        if paths2d.shape[1] < t:
            return np.array([], dtype=np.float64)
        ker = np.ones(t, dtype=np.float64)
        return np.concatenate([np.convolve(p, ker, mode="valid") for p in paths2d], axis=0)

    out = {}
    for t in t_lags:
        t = int(t)
        real_t = agg_returns_overlapping(real_r, t)
        fake_t = agg_over_paths(fake_paths, t)
        dy, nb = dy_metric(
            real_t, fake_t, dy_base_t=dy_base_t,
            mass_per_bin=5, alpha=1e-8, merge_empty=False,
            max_bins=100, return_bins=True
        )
        out[t] = {"DY": float(dy), "DY_bins": int(nb)}
    return out

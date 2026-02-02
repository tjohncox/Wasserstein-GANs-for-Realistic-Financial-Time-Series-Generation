# metrics.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ---------- TF metrics ----------
def acf_tf(x, lags):
    # x: (B,T,1)
    x = x[..., 0]
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)

    outs = []
    for k in range(1, int(lags) + 1):
        xk = x[:, :-k]
        yk = x[:,  k:]
        num = tf.reduce_sum(xk * yk, axis=1)
        den = tf.sqrt(
            (tf.reduce_sum(xk * xk, axis=1) + 1e-12) *
            (tf.reduce_sum(yk * yk, axis=1) + 1e-12)
        )
        outs.append(num / den)

    return tf.stack(outs, axis=1)  # (B,lags)


def leverage_tf(r, nlags=40):
    x = r[..., 0]
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)
    y = tf.square(r[..., 0])
    y = y - tf.reduce_mean(y, axis=1, keepdims=True)

    out = []
    for k in range(1, int(nlags) + 1):
        xk = x[:, :-k]
        yk = y[:,  k:]
        cov = tf.reduce_sum(xk * yk, axis=1, keepdims=True)
        x_var = tf.reduce_sum(tf.square(xk), axis=1, keepdims=True) + 1e-12
        y_var = tf.reduce_sum(tf.square(yk), axis=1, keepdims=True) + 1e-12
        corr = cov / tf.sqrt(x_var * y_var)
        out.append(corr)
    return tf.concat(out, axis=1)


def tf_kurtosis_per_batch(x):
    y = x[..., 0]
    m = tf.reduce_mean(y, axis=1, keepdims=True)
    v = tf.reduce_mean(tf.square(y - m), axis=1) + 1e-12
    c4 = tf.reduce_mean(tf.pow(y - m, 4), axis=1)
    k = c4 / (v * v)
    return tf.reduce_mean(k)


# ---------- numpy paper metrics ----------
def agg_returns_overlapping(r, t):
    r = np.asarray(r, dtype=np.float64)
    t = int(t)
    if t == 1:
        return r.copy()
    if len(r) < t:
        return np.array([], dtype=np.float64)
    return np.convolve(r, np.ones(t, dtype=np.float64), mode="valid")


def dy_metric(real_t, fake_t, dy_base_t, mass_per_bin=5, alpha=1e-8, merge_empty=False, max_bins=None, return_bins=False):
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


def acf_vec(x, S):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    S = int(S)
    out = np.zeros(S, dtype=np.float64)
    for k in range(1, S + 1):
        if k >= len(x):
            out[k - 1] = 0.0
            continue
        a = x[:-k]
        b = x[k:]
        num = np.dot(a, b)
        den = np.sqrt((np.dot(a, a) + 1e-12) * (np.dot(b, b) + 1e-12))
        out[k - 1] = num / den
    return out


def lev_vec(r, S):
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    S = int(S)
    x = r - r.mean()
    y = (r * r) - (r * r).mean()
    out = np.zeros(S, dtype=np.float64)
    for k in range(1, S + 1):
        if k >= len(r):
            out[k - 1] = 0.0
            continue
        a = x[:-k]
        b = y[k:]
        num = np.dot(a, b)
        den = np.sqrt((np.dot(a, a) + 1e-12) * (np.dot(b, b) + 1e-12))
        out[k - 1] = num / den
    return out


def paper_dependence_scores(real_r, fake_paths, S=250):
    real_r = np.asarray(real_r, dtype=np.float64)
    fake_paths = np.asarray(fake_paths, dtype=np.float64)
    S = int(S)

    C_real_x   = acf_vec(real_r, S)
    C_real_abs = acf_vec(np.abs(real_r), S)
    C_real_sq  = acf_vec(real_r * real_r, S)
    L_real     = lev_vec(real_r, S)

    C_fake_x   = np.mean([acf_vec(fake_paths[i], S) for i in range(fake_paths.shape[0])], axis=0)
    C_fake_abs = np.mean([acf_vec(np.abs(fake_paths[i]), S) for i in range(fake_paths.shape[0])], axis=0)
    C_fake_sq  = np.mean([acf_vec(fake_paths[i] * fake_paths[i], S) for i in range(fake_paths.shape[0])], axis=0)
    L_fake     = np.mean([lev_vec(fake_paths[i], S) for i in range(fake_paths.shape[0])], axis=0)

    scores = {
        "acf_x":  float(np.sqrt(np.mean((C_real_x  - C_fake_x )**2))),
        "acf_abs": float(np.sqrt(np.mean((C_real_abs - C_fake_abs)**2))),
        "acf_sq":  float(np.sqrt(np.mean((C_real_sq  - C_fake_sq )**2))),
        "lev":     float(np.sqrt(np.sum((L_real - L_fake)**2))),
    }
    return scores


def paper_distribution_metrics(real_r, fake_paths, dy_base_t, t_lags=(1, 5, 20, 100)):
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
        dy, nb = dy_metric(real_t, fake_t, dy_base_t=dy_base_t, mass_per_bin=5, alpha=1e-8, merge_empty=False, max_bins=100, return_bins=True)
        out[t] = {"DY": float(dy), "DY_bins": int(nb)}
    return out


class PaperEvaluator:
    def __init__(self, real_series, preproc, train_cfg, dy_base_t):
        self.real = np.asarray(real_series, dtype=np.float64)
        self.preproc = preproc
        self.cfg = train_cfg
        self.dy_base_t = int(dy_base_t)

        self.real_mean = float(np.mean(self.real))
        self.real_std = float(np.std(self.real) + 1e-12)
        self.real_qs = np.quantile(self.real, [0.01, 0.05, 0.95, 0.99]).astype(np.float64)

    def raw_stats(self, netG, z_dim, T_eval, burn_in, n_runs=3):
        pools = []
        for _ in range(int(n_runs)):
            z = tf.random.normal([1, int(T_eval) + int(burn_in), int(z_dim)])
            g_train = netG(z, training=False).numpy().reshape(-1)
            g_train = g_train[int(burn_in):int(burn_in) + int(T_eval)]
            pools.append(self.preproc.inverse_transform(g_train))
        pool = np.concatenate(pools, axis=0).astype(np.float64)
        return {
            "mean": float(np.mean(pool)),
            "std": float(np.std(pool) + 1e-12),
            "qs": np.quantile(pool, [0.01, 0.05, 0.95, 0.99]).astype(np.float64),
        }

    def sample_paths_raw(self, netG, z_dim, M, Ttilde, burn_in):
        outs = np.zeros((int(M), int(Ttilde)), dtype=np.float64)
        for i in range(int(M)):
            z = tf.random.normal([1, int(Ttilde) + int(burn_in), int(z_dim)])
            g_train = netG(z, training=False).numpy().reshape(-1)
            g_train = g_train[int(burn_in):int(burn_in) + int(Ttilde)]
            outs[i] = self.preproc.inverse_transform(g_train)
        return outs

    def paper_score(self, fake_paths):
        dep = paper_dependence_scores(self.real, fake_paths, S=self.cfg.sel_s)
        dist = paper_distribution_metrics(self.real, fake_paths, dy_base_t=self.dy_base_t, t_lags=self.cfg.paper_t_lags)

        dy_sum = 0.0
        dy_by_t = {}

        for t in self.cfg.paper_t_lags:
            t = int(t)
            dy_val = float(dist[t]["DY"])
            dy_by_t[t] = dy_val
            if not np.isnan(dy_val):
                dy_sum += dy_val

        scalar = (
            self.cfg.w_acf_x * float(dep["acf_x"])
            + self.cfg.w_acf_abs * float(dep["acf_abs"])
            + self.cfg.w_acf_sq * float(dep["acf_sq"])
            + self.cfg.w_lev * float(dep["lev"])
            + self.cfg.w_dy_sum * float(dy_sum)
        )

        parts = {
            "acf_x": float(dep["acf_x"]),
            "acf_abs": float(dep["acf_abs"]),
            "acf_sq": float(dep["acf_sq"]),
            "lev": float(dep["lev"]),
            "dy_sum": float(dy_sum),
            "dy_by_t": dy_by_t,
        }
        return float(scalar), parts


class Plotter:
    def __init__(self, show=True, save=False, out_dir=None):
        self.show = bool(show)
        self.save = bool(save)
        self.out_dir = out_dir

    def _finalize(self, fig, name):
        if self.save and self.out_dir and name:
            os.makedirs(self.out_dir, exist_ok=True)
            fig.savefig(os.path.join(self.out_dir, name), dpi=140, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _price_from_logret(P0, logret):
        return float(P0) * np.exp(np.cumsum(np.asarray(logret, dtype=np.float64)))

    def plot_price_paths(self, close_series, fake_logret_paths, n_paths=50, title="Price paths from log-returns", filename=None, alpha=0.35):
        close = close_series.dropna()
        dates = close.index[1:]
        real_price = close.iloc[1:].values
        P0 = float(close.iloc[0])

        n_paths = min(int(n_paths), int(fake_logret_paths.shape[0]))
        fig = plt.figure(figsize=(12, 5))
        for i in range(n_paths):
            fp = self._price_from_logret(P0, fake_logret_paths[i])
            plt.plot(dates, fp, linewidth=0.8, alpha=float(alpha))
        plt.plot(dates, real_price, linewidth=2.0, label="Real (Close)")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        self._finalize(fig, filename)

    def plot_acf_bundle(self, real_r, fake_paths, S=250, title_suffix="", filename_prefix=None):
        lags = np.arange(1, int(S) + 1)

        def _acf_stack(paths, transform=None):
            out = []
            for i in range(paths.shape[0]):
                x = paths[i]
                if transform is not None:
                    x = transform(x)
                out.append(acf_vec(x, int(S)))
            return np.stack(out, axis=0)

        # ACF(r)
        real_acf = acf_vec(real_r, int(S))
        fake_acf = _acf_stack(fake_paths)
        fake_mean = fake_acf.mean(axis=0)
        fake_lo, fake_hi = np.quantile(fake_acf, [0.05, 0.95], axis=0)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(lags, real_acf, linewidth=2.0, label="Real")
        plt.plot(lags, fake_mean, linewidth=2.0, label="Fake mean")
        plt.fill_between(lags, fake_lo, fake_hi, alpha=0.25, label="Fake 5-95%")
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"ACF of returns r (lags 1..{S}){title_suffix}")
        plt.xlabel("Lag k")
        plt.ylabel("ACF")
        plt.legend()
        plt.tight_layout()
        self._finalize(fig, f"{filename_prefix}_acf_r.png" if filename_prefix else None)

        # ACF(|r|)
        real_abs = acf_vec(np.abs(real_r), int(S))
        fake_abs = _acf_stack(fake_paths, transform=np.abs)
        abs_mean = fake_abs.mean(axis=0)
        abs_lo, abs_hi = np.quantile(fake_abs, [0.05, 0.95], axis=0)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(lags, real_abs, linewidth=2.0, label="Real")
        plt.plot(lags, abs_mean, linewidth=2.0, label="Fake mean")
        plt.fill_between(lags, abs_lo, abs_hi, alpha=0.25, label="Fake 5-95%")
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"ACF of |r| (volatility clustering){title_suffix}")
        plt.xlabel("Lag k")
        plt.ylabel("ACF")
        plt.legend()
        plt.tight_layout()
        self._finalize(fig, f"{filename_prefix}_acf_abs.png" if filename_prefix else None)

        # ACF(r^2)
        real_sq = acf_vec(real_r * real_r, int(S))
        fake_sq = _acf_stack(fake_paths, transform=lambda x: x * x)
        sq_mean = fake_sq.mean(axis=0)
        sq_lo, sq_hi = np.quantile(fake_sq, [0.05, 0.95], axis=0)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(lags, real_sq, linewidth=2.0, label="Real")
        plt.plot(lags, sq_mean, linewidth=2.0, label="Fake mean")
        plt.fill_between(lags, sq_lo, sq_hi, alpha=0.25, label="Fake 5-95%")
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"ACF of r^2 (volatility clustering){title_suffix}")
        plt.xlabel("Lag k")
        plt.ylabel("ACF")
        plt.legend()
        plt.tight_layout()
        self._finalize(fig, f"{filename_prefix}_acf_sq.png" if filename_prefix else None)

    def plot_leverage(self, real_r, fake_paths, S=250, title_suffix="", filename=None):
        lags = np.arange(1, int(S) + 1)

        real_lev = lev_vec(real_r, int(S))
        fake_lev = np.stack([lev_vec(fake_paths[i], int(S)) for i in range(fake_paths.shape[0])], axis=0)
        lev_mean = fake_lev.mean(axis=0)
        lev_lo, lev_hi = np.quantile(fake_lev, [0.05, 0.95], axis=0)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(lags, real_lev, linewidth=2.0, label="Real")
        plt.plot(lags, lev_mean, linewidth=2.0, label="Fake mean")
        plt.fill_between(lags, lev_lo, lev_hi, alpha=0.25, label="Fake 5-95%")
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"Leverage corr(r_t, r^2_(t+k)) (lags 1..{S}){title_suffix}")
        plt.xlabel("Lag k")
        plt.ylabel("Correlation")
        plt.legend()
        plt.tight_layout()
        self._finalize(fig, filename)

"""Preprocessing utilities for financial time series."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import kurtosis
from scipy.special import lambertw
from scipy.optimize import minimize


# --- Delta init settings (only for initialization) ---
DELTA_MIN = 1e-6
DELTA_INIT_MAX = 0.24   # kurtosis-based Taylor init is conceptually for delta < 0.25
DELTA_MLE_MAX = 10.0    # optimization upper bound (can be larger; keep stable)


def log_returns_from_close(df, close_col="Close"):
    close = df[close_col].dropna()
    logret = np.log(close / close.shift(1)).dropna().values.astype(np.float64)
    if len(logret) <= 0:
        raise ValueError("[DATA] logret is empty.")
    return logret


def lambertw_forward_heavytail(x, delta):
    """Forward transformation for Lambert-W heavy-tail distribution."""
    x = np.asarray(x, dtype=np.float64)
    if delta <= 0:
        return x
    exp_arg = 0.5 * delta * (x * x)
    exp_arg = np.minimum(exp_arg, 700.0)  # avoid overflow
    return x * np.exp(exp_arg)


def W_delta(z, delta):
    """Inverse Lambert-W transformation."""
    z = np.asarray(z, dtype=np.float64)
    delta = float(delta)
    if delta <= 0:
        return z
    arg = delta * (z ** 2)
    arg = np.minimum(arg, 1e12)  # numerical safety
    w = lambertw(arg).real
    return np.sign(z) * np.sqrt(np.maximum(w, 0.0) / (delta + 1e-18))

def delta_init_taylor(z, delta_min=DELTA_MIN, delta_max=DELTA_INIT_MAX):
    """
    Kurtosis-based Taylor init for delta:
      delta0 = (sqrt(66*k - 162) - 6) / 66
    with safeguards if 66*k - 162 <= 0.
    """
    z = np.asarray(z, dtype=np.float64)
    k = float(kurtosis(z, fisher=False, bias=False))
    inside = 66.0 * k - 162.0
    if (not np.isfinite(inside)) or inside <= 0:
        return float(delta_min)
    d0 = (np.sqrt(inside) - 6.0) / 66.0
    return float(np.clip(d0, delta_min, delta_max))


def _lambertw_negloglik(params, y):
    """Negative log-likelihood for Lambert-W distribution."""
    mu, log_sigma, log_delta = params
    sigma = float(np.exp(log_sigma))
    delta = float(np.exp(log_delta))

    z = (y - mu) / (sigma + 1e-18)
    x = W_delta(z, delta)

    log_phi = -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)
    log_jac = -np.log(sigma) - 0.5 * delta * x * x - np.log1p(delta * x * x)

    nll = -np.sum(log_phi + log_jac)
    return nll if np.isfinite(nll) else 1e20


def lambertw_mle(y):
    """
    Maximum likelihood estimation for Lambert-W parameters.
    Returns (mu, sigma, delta).
    """
    y = np.asarray(y, dtype=np.float64)
    mu0 = float(np.mean(y))
    sigma0 = float(np.std(y) + 1e-12)

    # delta init 
    delta0 = float(delta_init_taylor(y))
    delta0 = max(delta0, DELTA_MIN)

    x0 = np.array([mu0, np.log(sigma0), np.log(delta0)], dtype=np.float64)
    bounds = [
        (None, None),
        (np.log(1e-6), np.log(1e3)),
        (np.log(DELTA_MIN), np.log(DELTA_MLE_MAX)),
    ]

    res = minimize(_lambertw_negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds)

    # Robust fallback if optimizer fails
    if (not res.success) or (not np.all(np.isfinite(res.x))):
        return mu0, sigma0, delta0

    mu, log_sigma, log_delta = res.x
    return float(mu), float(np.exp(log_sigma)), float(np.exp(log_delta))


@dataclass
class LambertWState:
    """State for Lambert-W transformation."""
    r_mean: float
    r_std: float
    lam_mu: float = 0.0
    lam_sigma: float = 1.0
    delta_hat: float = 0.0
    u_mean: float = 0.0
    u_std: float = 1.0


class LambertWPreprocessor:
    """Lambert-W preprocessor for heavy-tailed distributions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = None

    def fit(self, logret):
        logret = np.asarray(logret, dtype=np.float64).reshape(-1)
        r_mean = float(np.mean(logret))
        r_std = float(np.std(logret) + 1e-12)

        r_norm = (logret - r_mean) / r_std
        st = LambertWState(r_mean=r_mean, r_std=r_std)

        if self.cfg.use_lambert:
            lam_mu, lam_sigma, delta_hat = lambertw_mle(r_norm)
            z = (r_norm - lam_mu) / (lam_sigma + 1e-18)
            u = W_delta(z, delta_hat)

            st.lam_mu = lam_mu
            st.lam_sigma = lam_sigma
            st.delta_hat = delta_hat
            st.u_mean = float(np.mean(u))
            st.u_std = float(np.std(u) + 1e-12)

        self.state = st
        return self

    def transform(self, logret):
        assert self.state is not None, "Call fit() first."
        st = self.state

        logret = np.asarray(logret, dtype=np.float64).reshape(-1)
        r_norm = (logret - st.r_mean) / st.r_std

        if self.cfg.use_lambert:
            z = (r_norm - st.lam_mu) / (st.lam_sigma + 1e-18)
            u = W_delta(z, st.delta_hat)
            if self.cfg.renorm_after_lambert:
                r_proc = (u - st.u_mean) / st.u_std
            else:
                r_proc = u
        else:
            r_proc = r_norm.copy()

        return r_proc.astype(np.float32)

    def inverse_transform(self, r_train_hat):
        """
        Inverse transform processed returns back to original scale.
        """
        assert self.state is not None, "Call fit() first."
        st = self.state
        r_proc_hat = np.asarray(r_train_hat, dtype=np.float64)

        if self.cfg.use_lambert:
            if self.cfg.renorm_after_lambert:
                u_hat = r_proc_hat * st.u_std + st.u_mean
            else:
                u_hat = r_proc_hat

            r_norm_hat = st.lam_mu + st.lam_sigma * lambertw_forward_heavytail(u_hat, st.delta_hat)
        else:
            r_norm_hat = r_proc_hat

        r_hat = r_norm_hat * st.r_std + st.r_mean
        return r_hat.astype(np.float64)

    def summary(self):
        assert self.state is not None, "Call fit() first."
        st = self.state
        return {
            "use_lambert": bool(self.cfg.use_lambert),
            "renorm_after_lambert": bool(self.cfg.renorm_after_lambert),
            "r_mean": st.r_mean,
            "r_std": st.r_std,
            "lam_mu": st.lam_mu,
            "lam_sigma": st.lam_sigma,
            "delta_hat": st.delta_hat,
            "u_mean": st.u_mean,
            "u_std": st.u_std,
        }

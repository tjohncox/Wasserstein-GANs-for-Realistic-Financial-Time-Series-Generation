"""Evaluator for QuantGAN models using paper metrics."""

import numpy as np
import tensorflow as tf
from quantgan.evaluation.metrics import (
    paper_dependence_scores,
    paper_distribution_metrics,
)


class PaperEvaluator:
    """Evaluator using metrics from the QuantGAN paper."""

    def __init__(self, real_series, preproc, train_cfg, dy_base_t):
        """Initialize evaluator.
        
        Args:
            real_series: Real log returns (1D array)
            preproc: Preprocessor with inverse_transform method
            train_cfg: TrainConfig instance
            dy_base_t: Base period for DY metric
        """
        self.real = np.asarray(real_series, dtype=np.float64)
        self.preproc = preproc
        self.cfg = train_cfg
        self.dy_base_t = int(dy_base_t)

        self.real_mean = float(np.mean(self.real))
        self.real_std = float(np.std(self.real) + 1e-12)
        self.real_qs = np.quantile(self.real, [0.01, 0.05, 0.95, 0.99]).astype(
            np.float64
        )

    def raw_stats(self, netG, z_dim, T_eval, burn_in, n_runs=3, batch=50, seed=None):
        """Compute quick sanity stats in raw space.
        
        Args:
            netG: Generator model
            z_dim: Latent dimension
            T_eval: Evaluation length
            burn_in: Burn-in length
            n_runs: Number of paths
            batch: Batch size
            seed: Random seed
            
        Returns:
            Dictionary with mean, std, and quantiles
        """
        paths = self.sample_paths_raw(
            netG=netG,
            z_dim=z_dim,
            M=int(n_runs),
            Ttilde=int(T_eval),
            burn_in=int(burn_in),
            batch=int(batch),
            seed=seed,
        )
        pool = np.asarray(paths, dtype=np.float64).reshape(-1)
        return {
            "mean": float(np.mean(pool)),
            "std": float(np.std(pool) + 1e-12),
            "qs": np.quantile(pool, [0.01, 0.05, 0.95, 0.99]).astype(np.float64),
        }

    def sample_paths_raw(
        self, netG, z_dim, M, Ttilde, burn_in, batch=50, seed=None
    ):
        """Generate M raw log-return paths.
        
        Args:
            netG: Generator model
            z_dim: Latent dimension
            M: Number of paths
            Ttilde: Length of each path
            burn_in: Burn-in length
            batch: Batch size
            seed: Random seed
            
        Returns:
            Array of shape (M, Ttilde) with raw log-returns
        """
        M = int(M)
        Ttilde = int(Ttilde)
        burn_in = int(burn_in)
        z_dim = int(z_dim)
        batch = int(batch)

        outs = np.zeros((M, Ttilde), dtype=np.float64)
        done = 0
        g = (
            tf.random.Generator.from_seed(int(seed))
            if seed is not None
            else None
        )

        while done < M:
            b = min(batch, M - done)
            if g is None:
                z = tf.random.normal([b, Ttilde + burn_in, z_dim])
            else:
                z = g.normal([b, Ttilde + burn_in, z_dim])

            fake_full = netG(z, training=False).numpy()[..., 0]
            fake_used = fake_full[:, burn_in:burn_in + Ttilde]
            outs[done:done + b] = self.preproc.inverse_transform(fake_used)
            done += b

        return outs

    def paper_score(self, fake_paths):
        """Compute paper score from generated paths.
        
        Args:
            fake_paths: Generated paths (2D array: M x T)
            
        Returns:
            Tuple of (score, parts_dict)
        """
        dep = paper_dependence_scores(
            self.real, fake_paths, max_lags=self.cfg.sel_s
        )
        dist = paper_distribution_metrics(
            self.real,
            fake_paths,
            dy_base_t=self.dy_base_t,
            t_lags=self.cfg.paper_t_lags,
        )

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

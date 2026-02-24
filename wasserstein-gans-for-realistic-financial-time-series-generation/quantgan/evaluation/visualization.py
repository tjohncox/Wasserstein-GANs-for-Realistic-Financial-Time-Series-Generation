"""Visualization utilities for QuantGAN results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from quantgan.evaluation.metrics import acf_vec, lev_vec, agg_returns_overlapping


class Plotter:
    """Plotter for visualizing real and generated time series."""

    def __init__(self, show=True, save=False, out_dir=None):
        """Initialize plotter.
        
        Args:
            show: Show plots interactively
            save: Save plots to disk
            out_dir: Output directory for saved plots
        """
        self.show = bool(show)
        self.save = bool(save)
        self.out_dir = out_dir

    def _finalize(self, fig, name):
        """Finalize plot: save and/or show.
        
        Args:
            fig: Matplotlib figure
            name: Filename for saving
        """
        if self.save and self.out_dir and name:
            os.makedirs(self.out_dir, exist_ok=True)
            fig.savefig(
                os.path.join(self.out_dir, name), dpi=140, bbox_inches="tight"
            )
        if self.show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _price_from_logret(P0, logret):
        """Compute price from initial price and log returns.
        
        Args:
            P0: Initial price
            logret: Log returns
            
        Returns:
            Price series
        """
        return float(P0) * np.exp(np.cumsum(np.asarray(logret, dtype=np.float64)))

    def plot_price_paths(
        self,
        close_series,
        fake_logret_paths,
        n_paths=50,
        title="Price paths from log-returns",
        filename=None,
        alpha=0.35,
    ):
        """Plot real price and synthetic price paths.
        
        Args:
            close_series: Real close prices (pandas Series with DatetimeIndex)
            fake_logret_paths: Generated log returns (2D array)
            n_paths: Number of paths to plot
            title: Plot title
            filename: Filename for saving
            alpha: Alpha for fake paths
        """
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

    def plot_acf_bundle(
        self, real_r, fake_paths, S=250, title_suffix="", filename_prefix=None
    ):
        """Plot ACF for returns, absolute returns, and squared returns.
        
        Args:
            real_r: Real returns (1D)
            fake_paths: Fake returns (2D)
            S: Number of lags
            title_suffix: Suffix for titles
            filename_prefix: Prefix for filenames
        """
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
        self._finalize(
            fig, f"{filename_prefix}_acf_r.png" if filename_prefix else None
        )

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
        self._finalize(
            fig, f"{filename_prefix}_acf_abs.png" if filename_prefix else None
        )

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
        self._finalize(
            fig, f"{filename_prefix}_acf_sq.png" if filename_prefix else None
        )

    def plot_leverage(
        self, real_r, fake_paths, S=250, title_suffix="", filename=None
    ):
        """Plot leverage correlation.
        
        Args:
            real_r: Real returns (1D)
            fake_paths: Fake returns (2D)
            S: Number of lags
            title_suffix: Suffix for title
            filename: Filename for saving
        """
        lags = np.arange(1, int(S) + 1)

        real_lev = lev_vec(real_r, int(S))
        fake_lev = np.stack(
            [lev_vec(fake_paths[i], int(S)) for i in range(fake_paths.shape[0])],
            axis=0,
        )
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

    def plot_hist_panel(
        self,
        real_r,
        fake_paths,
        t_lags=(1, 5, 20, 100),
        bins=120,
        clip_q=(0.001, 0.999),
        title="Histogram panel (Real vs Fake)",
        filename=None,
        label_fake="Fake",
        label_real="Real",
    ):
        """Plot histogram panel for aggregated returns.
        
        Args:
            real_r: Real returns (1D)
            fake_paths: Fake returns (2D)
            t_lags: Aggregation periods
            bins: Number of bins
            clip_q: Quantiles for clipping
            title: Overall title
            filename: Filename for saving
            label_fake: Label for fake data
            label_real: Label for real data
        """
        real_r = np.asarray(real_r, dtype=np.float64).reshape(-1)
        fake_paths = np.asarray(fake_paths, dtype=np.float64)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.flatten()

        letters = ["(a)", "(b)", "(c)", "(d)"]
        names = {1: "1d", 5: "5d", 20: "20d", 100: "100d", 200: "200d"}

        qlo, qhi = clip_q

        for ax, t, letter in zip(axes, t_lags, letters):
            t = int(t)

            real_t = agg_returns_overlapping(real_r, t)
            fake_t = np.concatenate(
                [
                    agg_returns_overlapping(fake_paths[i], t)
                    for i in range(fake_paths.shape[0])
                ],
                axis=0,
            )

            if len(real_t) < 10 or len(fake_t) < 10:
                ax.set_visible(False)
                continue

            # Robust x-range (clipping)
            lo, hi = np.quantile(real_t, [qlo, qhi])
            lo = float(lo)
            hi = float(hi)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(min(np.min(real_t), np.min(fake_t)))
                hi = float(max(np.max(real_t), np.max(fake_t)))
                if hi <= lo:
                    lo -= 1e-6
                    hi += 1e-6

            pad = 0.03 * (hi - lo + 1e-12)
            edges = np.linspace(lo - pad, hi + pad, int(bins) + 1)

            ax.hist(fake_t, bins=edges, density=True, alpha=0.75, label=label_fake)
            ax.hist(real_t, bins=edges, density=True, alpha=0.75, label=label_real)

            ax.set_title(f"{letter} {names.get(t, f'{t}d')}")
            ax.set_xlabel("Aggregated log return")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

        fig.suptitle(title)
        fig.tight_layout()

        self._finalize(fig, filename)

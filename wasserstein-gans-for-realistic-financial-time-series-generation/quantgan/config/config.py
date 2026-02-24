"""Configuration classes for QuantGAN."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Configuration for data loading."""
    ticker: str = "SPY"
    start: str = "2009-05-01"
    end: str = "2018-12-31"
    interval: str = "1d"
    source: str = "defeatbeta"  # "defeatbeta" or "yfinance"


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    use_lambert: bool = True
    renorm_after_lambert: bool = True
    scale_q: float = 0.999


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    window_len: int = 127
    batch_size: int = 30
    weighted_sampling: bool = True
    seed: int = 7


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    generator_type: str = "pure_tcn"  # "svnn" or "pure_tcn"
    causal: bool = True
    z_dim: int = 3
    kernel: int = 2
    dilations: Tuple[int, ...] = (1, 1, 2, 4, 8, 16, 32)
    use_skip_connections: bool = True

    # Generator
    g_ch: int = 80
    g_ch_hidden: int = 80
    g_use_layernorm: bool = True
    g_weight_decay: float = 1e-4
    g_constrained_innovation: bool = True
    g_eps_hidden: int = 16 # relevant for constrained_innovation = False
    g_sigma_mode: str = "softplus"  # "abs" or "softplus", relevant for SVNN
    g_use_soft_clip: bool = True
    g_soft_clip_c: float = 4.0 # relevant for g_use_soft_clip = True

    # Discriminator
    d_ch: int = 80
    d_ch_hidden: int = 80
    d_use_layernorm: bool = False
    d_weight_decay: float = 0.0


@dataclass
class TrainConfig:
    """Configuration for training."""
    epochs: int = 200
    n_critic: int = 5
    lambda_gp: float = 10.0
    pretrain_d_epochs: int = 5

    lr_d0: float = 5e-5
    lr_g0: float = 1e-4
    decay_start: int = 100
    decay_rate: float = 0.98
    min_lr: float = 1e-6

    sel_every: int = 5
    sel_m: int = 500
    sel_ttilde: int = 4000
    sel_s: int = 250
    paper_t_lags: Tuple[int, ...] = (1, 5, 20, 100)

    w_acf_x: float = 1.0
    w_acf_abs: float = 1.0
    w_acf_sq: float = 1.0
    w_lev: float = 0.1
    w_dy_sum: float = 1 / 4000


@dataclass
class DebugConfig:
    """Configuration for debugging."""
    verbose: bool = True
    debug_tails: bool = True
    debug_raw_every: int = 5


@dataclass
class RunConfig:
    """Configuration for run settings."""
    out_dir: str = "quantgan_outputs_wgangp"
    seed: int = 0
    vol_lags: int = 40
    lev_lags: int = 40
    n_target_windows: int = 4000
    n_plot_runs: int = 50
    n_plot_paths: int = 50
    show_plots: bool = True
    save_plots: bool = False

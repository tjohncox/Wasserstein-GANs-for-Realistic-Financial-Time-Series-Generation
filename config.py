# config.py
from dataclasses import dataclass


@dataclass
class DataConfig:
    ticker = "SPY"
    start = "2009-05-01"
    end = "2018-12-31"
    interval = "1d"


@dataclass
class PreprocessConfig:
    use_lambert = True
    renorm_after_lambert = True
    scale_q = 0.999


@dataclass
class DatasetConfig:
    window_len = 127
    batch_size = 30
    weighted_sampling = True
    seed = 7


@dataclass
class ModelConfig:
    generator_type = "pure_tcn"   # "svnn" oder "pure_tcn"
    causal = True
    z_dim = 3
    kernel = 2
    dilations = (1, 1, 2, 4, 8, 16, 32)
    use_skip_connections = True

    # Generator
    g_ch = 50
    g_ch_hidden = 50
    g_use_layernorm = True
    g_weight_decay = 1e-4
    g_constrained_innovation = True
    g_eps_hidden = 16
    g_sigma_mode = "abs"   # "abs" or "softplus"
    g_use_soft_clip = False
    g_soft_clip_c = 4.0

    # Discriminator
    d_ch = 80
    d_ch_hidden = 80
    d_use_layernorm = False
    d_weight_decay = 0.0


@dataclass
class TrainConfig:
    epochs = 250
    n_critic = 5
    lambda_gp = 10.0
    pretrain_d_epochs = 5

    lr_d0 = 5e-5
    lr_g0 = 1e-4
    decay_start = 100
    decay_rate = 0.98
    min_lr = 1e-6

    sel_every = 5
    sel_m = 500
    sel_ttilde = 4000
    sel_s = 250
    paper_t_lags = (1, 5, 20, 100)

    w_acf_x = 1.0
    w_acf_abs = 1.0
    w_acf_sq = 1.0
    w_lev = 0.1
    w_emd_sum = 1.0
    w_dy_sum = 1 / 4000


@dataclass
class DebugConfig:
    verbose = True
    debug_tails = True
    debug_raw_every = 5


@dataclass
class RunConfig:
    out_dir = "quantgan_outputs_wgangp"
    seed = 0
    vol_lags = 40
    lev_lags = 40
    n_target_windows = 4000
    n_plot_runs = 50
    n_plot_paths = 50
    show_plots = True
    save_plots = False

"""Model registry and factory."""

from quantgan.models.generator import build_G_svnn, build_G_pure_tcn
from quantgan.models.discriminator import build_D


def build_generator(model_cfg):
    """Build generator based on model_cfg.generator_type.
    
    Supported types:
      - svnn: Stochastic Volatility Neural Network
      - pure_tcn: Pure Temporal Convolutional Network
      
    Args:
        model_cfg: ModelConfig instance
        
    Returns:
        Keras Model (generator)
    """
    GENERATOR_BUILDERS = {
        "svnn": lambda cfg: build_G_svnn(
            z_dim=cfg.z_dim,
            ch=cfg.g_ch,
            ch_hidden=cfg.g_ch_hidden,
            kernel=cfg.kernel,
            dilations=cfg.dilations,
            causal=cfg.causal,
            use_skips=cfg.use_skip_connections,
            use_soft_clip=cfg.g_use_soft_clip,
            soft_clip_c=cfg.g_soft_clip_c,
            constrained_innovation=getattr(cfg, "g_constrained_innovation", True),
            eps_hidden=getattr(cfg, "g_eps_hidden", 16),
            use_layernorm=getattr(cfg, "g_use_layernorm", False),
            weight_decay=getattr(cfg, "g_weight_decay", 0.0),
            sigma_mode=getattr(cfg, "g_sigma_mode", "abs"),
        ),
        "pure_tcn": lambda cfg: build_G_pure_tcn(
            z_dim=cfg.z_dim,
            ch=cfg.g_ch,
            ch_hidden=cfg.g_ch_hidden,
            kernel=cfg.kernel,
            dilations=cfg.dilations,
            causal=cfg.causal,
            use_skips=cfg.use_skip_connections,
            use_soft_clip=cfg.g_use_soft_clip,
            soft_clip_c=cfg.g_soft_clip_c,
            use_layernorm=getattr(cfg, "g_use_layernorm", False),
            weight_decay=getattr(cfg, "g_weight_decay", 0.0),
        ),
    }

    gen_type = str(getattr(model_cfg, "generator_type", "svnn")).lower().strip()
    if gen_type not in GENERATOR_BUILDERS:
        raise ValueError(
            f"Unknown generator_type={gen_type}. "
            f"Allowed={list(GENERATOR_BUILDERS.keys())}"
        )
    return GENERATOR_BUILDERS[gen_type](model_cfg)


def build_discriminator(model_cfg):
    """Build discriminator.
    
    Args:
        model_cfg: ModelConfig instance
        
    Returns:
        Keras Model (discriminator)
    """
    return build_D(
        ch=model_cfg.d_ch,
        ch_hidden=model_cfg.d_ch_hidden,
        kernel=model_cfg.kernel,
        dilations=model_cfg.dilations,
        causal=model_cfg.causal,
        use_skips=model_cfg.use_skip_connections,
        use_layernorm=getattr(model_cfg, "d_use_layernorm", False),
        weight_decay=getattr(model_cfg, "d_weight_decay", 0.0),
    )

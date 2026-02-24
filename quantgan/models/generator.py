"""Generator architectures for QuantGAN."""

import tensorflow as tf
from quantgan.models.blocks import TCNBlock, l2_reg, aggregate_skip_connections


def build_G_pure_tcn(
    z_dim,
    ch,
    ch_hidden,
    kernel,
    dilations,
    causal=True,
    use_skips=True,
    use_soft_clip=False,
    soft_clip_c=3.0,
    use_layernorm=False,
    weight_decay=0.0,
):
    """Build pure TCN generator.
    
    Args:
        z_dim: Latent dimension
        ch: Number of channels
        ch_hidden: Hidden channels
        kernel: Kernel size
        dilations: Tuple of dilation rates
        causal: Use causal convolutions
        use_skips: Use skip connections
        use_soft_clip: Apply soft clipping
        soft_clip_c: Clipping parameter
        use_layernorm: Use layer normalization
        weight_decay: L2 regularization strength
        
    Returns:
        Keras Model
    """
    z_in = tf.keras.Input(shape=(None, z_dim))
    x = z_in
    skips = []

    for i, dilation in enumerate(dilations):
        k = 1 if i == 0 else kernel
        x = TCNBlock(
            ch_out=ch,
            kernel=k,
            dilation=dilation,
            causal=causal,
            ch_hidden=ch_hidden,
            use_layernorm=use_layernorm,
            weight_decay=weight_decay,
        )(x)

        if use_skips:
            reg = l2_reg(weight_decay)
            skips.append(
                tf.keras.layers.Conv1D(ch, 1, padding="same", kernel_regularizer=reg)(x)
            )

    # Aggregate skip connections
    skip_output = aggregate_skip_connections(skips)
    if skip_output is not None:
        x = skip_output

    reg_out = l2_reg(weight_decay)
    x = tf.keras.layers.Conv1D(1, 1, padding="same", kernel_regularizer=reg_out)(x)

    if use_soft_clip:
        c = float(soft_clip_c)
        x = tf.keras.layers.Lambda(lambda t: c * tf.tanh(t / c), name="soft_clip")(x)

    return tf.keras.Model(z_in, x, name="G_PURE_TCN")


def build_G_svnn(
    z_dim,
    ch,
    ch_hidden,
    kernel,
    dilations,
    causal=True,
    use_skips=True,
    use_soft_clip=False,
    soft_clip_c=3.0,
    constrained_innovation=True,
    eps_hidden=16,
    use_layernorm=False,
    weight_decay=0.0,
    sigma_mode="abs",  # "abs" (paper) or "softplus"
):
    """Build SVNN (Stochastic Volatility Neural Network) generator.
    
    This generator models returns as: r_t = sigma_t * eps_t + mu_t
    where sigma_t and mu_t are predicted by a TCN based on z_{<=t-1},
    and eps_t is the innovation from z_t.
    
    Args:
        z_dim: Latent dimension
        ch: Number of channels
        ch_hidden: Hidden channels
        kernel: Kernel size
        dilations: Tuple of dilation rates
        causal: Use causal convolutions
        use_skips: Use skip connections
        use_soft_clip: Apply soft clipping
        soft_clip_c: Clipping parameter
        constrained_innovation: Use first latent dimension as innovation
        eps_hidden: Hidden units for innovation MLP (if not constrained)
        use_layernorm: Use layer normalization
        weight_decay: L2 regularization strength
        sigma_mode: "abs" or "softplus" for sigma activation
        
    Returns:
        Keras Model
    """
    z_in = tf.keras.Input(shape=(None, z_dim))

    # Shift z by 1 for sigma/mu TCN -> depends on z_{<=t-1}
    z_shift = tf.keras.layers.Lambda(
        lambda z: tf.pad(z[:, :-1, :], [[0, 0], [1, 0], [0, 0]]),
        name="z_shift_for_sigma_mu"
    )(z_in)

    x = z_shift
    skips = []

    for i, dilation in enumerate(dilations):
        k = 1 if i == 0 else kernel
        x = TCNBlock(
            ch_out=ch,
            kernel=k,
            dilation=dilation,
            causal=causal,
            ch_hidden=ch_hidden,
            use_layernorm=use_layernorm,
            weight_decay=weight_decay,
        )(x)

        if use_skips:
            reg = l2_reg(weight_decay)
            skips.append(
                tf.keras.layers.Conv1D(ch, 1, padding="same", kernel_regularizer=reg)(x)
            )

    # Aggregate skip connections
    skip_output = aggregate_skip_connections(skips)
    if skip_output is not None:
        x = skip_output

    # Project to [sigma_raw, mu]
    reg_out = l2_reg(weight_decay)
    h = tf.keras.layers.Conv1D(
        2, 1, padding="same", kernel_regularizer=reg_out, name="sigma_mu_head"
    )(x)

    sigma_raw = tf.keras.layers.Lambda(lambda t: t[..., 0:1], name="sigma_raw")(h)
    mu = tf.keras.layers.Lambda(lambda t: t[..., 1:2], name="mu")(h)

    if sigma_mode == "softplus":
        sigma = tf.keras.layers.Activation(tf.nn.softplus, name="sigma_softplus")(
            sigma_raw
        )
    else:
        sigma = tf.keras.layers.Lambda(
            lambda t: tf.abs(t) + 1e-6, name="sigma_abs"
        )(sigma_raw)

    # Innovation eps_t
    if constrained_innovation:
        eps = tf.keras.layers.Lambda(
            lambda z: z[..., 0:1], name="eps_constrained"
        )(z_in)
    else:
        e = tf.keras.layers.Dense(eps_hidden, activation="relu")(z_in)
        eps = tf.keras.layers.Dense(1, activation=None, name="eps_mlp")(e)

    # Linear output (no tanh)
    r = tf.keras.layers.Lambda(
        lambda xs: xs[0] * xs[1] + xs[2], name="svnn_return"
    )([sigma, eps, mu])

    if use_soft_clip:
        c = float(soft_clip_c)
        r = tf.keras.layers.Lambda(lambda t: c * tf.tanh(t / c), name="soft_clip")(r)

    return tf.keras.Model(z_in, r, name="G_SVNN")

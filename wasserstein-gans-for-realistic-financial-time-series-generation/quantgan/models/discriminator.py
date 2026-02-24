"""Discriminator architecture for QuantGAN."""

import tensorflow as tf
from quantgan.models.blocks import TCNBlock, l2_reg, aggregate_skip_connections


def build_D(
    ch,
    ch_hidden,
    kernel,
    dilations,
    causal=True,
    use_skips=True,
    use_layernorm=False,
    weight_decay=0.0,
):
    """Build discriminator network.
    
    Args:
        ch: Number of channels
        ch_hidden: Hidden channels
        kernel: Kernel size
        dilations: Tuple of dilation rates
        causal: Use causal convolutions
        use_skips: Use skip connections
        use_layernorm: Use layer normalization
        weight_decay: L2 regularization strength
        
    Returns:
        Keras Model
    """
    x_in = tf.keras.Input(shape=(None, 1))
    x = x_in
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

    # Only last timestep
    x = tf.keras.layers.Lambda(lambda t: t[:, -1:, :])(x)

    reg_out = l2_reg(weight_decay)
    x = tf.keras.layers.Conv1D(1, 1, padding="same", kernel_regularizer=reg_out)(x)
    x = tf.keras.layers.Flatten()(x)
    
    return tf.keras.Model(x_in, x, name="D")

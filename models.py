# models.py
import tensorflow as tf


def conv1d(filters, kernel, dilation, causal=True, weight_decay=0.0):
    padding = "causal" if causal else "same"
    reg = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
    return tf.keras.layers.Conv1D(
        filters=int(filters),
        kernel_size=int(kernel),
        dilation_rate=int(dilation),
        padding=padding,
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
    )


class TCNBlock(tf.keras.layers.Layer):
    """
    PReLU(Conv(PReLU(Conv(x))))
    Optional: LayerNorm nach Conv (vor PReLU).
    """
    def __init__(self, ch_out, kernel, dilation, causal=True, ch_hidden=None,
                 use_layernorm=False, weight_decay=0.0):
        super().__init__()
        self.ch_out = int(ch_out)
        self.ch_hidden = int(ch_hidden) if ch_hidden is not None else int(ch_out)
        self.use_layernorm = bool(use_layernorm)

        self.conv1 = conv1d(self.ch_hidden, kernel, dilation, causal, weight_decay=weight_decay)
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-3, scale=True) if self.use_layernorm else None
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1])

        self.conv2 = conv1d(self.ch_out, kernel, dilation, causal, weight_decay=weight_decay)
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-3, scale=True) if self.use_layernorm else None
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1])

    def call(self, x, training=False):
        y = self.conv1(x)
        if self.ln1 is not None:
            y = self.ln1(y, training=training)
        y = self.prelu1(y)

        y = self.conv2(y)
        if self.ln2 is not None:
            y = self.ln2(y, training=training)
        y = self.prelu2(y)
        return y


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
    z_in = tf.keras.Input(shape=(None, int(z_dim)))  # (B,T,Z)
    x = z_in
    skips = []

    for i, d in enumerate(dilations):
        k = 1 if i == 0 else int(kernel)
        x = TCNBlock(
            ch_out=ch,
            kernel=k,
            dilation=int(d),
            causal=causal,
            ch_hidden=ch_hidden,
            use_layernorm=use_layernorm,
            weight_decay=weight_decay,
        )(x)

        if use_skips:
            reg = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
            skips.append(tf.keras.layers.Conv1D(ch, 1, padding="same", kernel_regularizer=reg)(x))

    if use_skips and len(skips) > 0:
        x = tf.keras.layers.Add()(skips) if len(skips) > 1 else skips[0]

    reg_out = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
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
    sigma_mode="abs",   # "abs" (paper) or "softplus"
):
    z_in = tf.keras.Input(shape=(None, int(z_dim)))  # (B,T,Z)

    # shift z by 1 for sigma/mu TCN -> depends on z_{<=t-1}
    z_shift = tf.keras.layers.Lambda(
        lambda z: tf.pad(z[:, :-1, :], [[0, 0], [1, 0], [0, 0]]),
        name="z_shift_for_sigma_mu"
    )(z_in)

    x = z_shift
    skips = []

    for i, d in enumerate(dilations):
        k = 1 if i == 0 else int(kernel)
        x = TCNBlock(
            ch_out=ch,
            kernel=k,
            dilation=int(d),
            causal=causal,
            ch_hidden=ch_hidden,
            use_layernorm=use_layernorm,
            weight_decay=weight_decay,
        )(x)

        if use_skips:
            reg = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
            skips.append(tf.keras.layers.Conv1D(ch, 1, padding="same", kernel_regularizer=reg)(x))

    if use_skips and len(skips) > 0:
        x = tf.keras.layers.Add()(skips) if len(skips) > 1 else skips[0]

    # project to [sigma_raw, mu]
    reg_out = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
    h = tf.keras.layers.Conv1D(2, 1, padding="same", kernel_regularizer=reg_out, name="sigma_mu_head")(x)

    sigma_raw = tf.keras.layers.Lambda(lambda t: t[..., 0:1], name="sigma_raw")(h)
    mu = tf.keras.layers.Lambda(lambda t: t[..., 1:2], name="mu")(h)

    if sigma_mode == "softplus":
        sigma = tf.keras.layers.Activation(tf.nn.softplus, name="sigma_softplus")(sigma_raw)
    else:
        sigma = tf.keras.layers.Lambda(lambda t: tf.abs(t) + 1e-6, name="sigma_abs")(sigma_raw)

    # innovation eps_t
    if constrained_innovation:
        eps = tf.keras.layers.Lambda(lambda z: z[..., 0:1], name="eps_constrained")(z_in)
    else:
        e = tf.keras.layers.Dense(int(eps_hidden), activation="relu")(z_in)
        eps = tf.keras.layers.Dense(1, activation=None, name="eps_mlp")(e)

    # linear output (kein tanh)
    r = tf.keras.layers.Lambda(lambda xs: xs[0] * xs[1] + xs[2], name="svnn_return")([sigma, eps, mu])

    if use_soft_clip:
        c = float(soft_clip_c)
        r = tf.keras.layers.Lambda(lambda t: c * tf.tanh(t / c), name="soft_clip")(r)

    return tf.keras.Model(z_in, r, name="G_SVNN")


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
    x_in = tf.keras.Input(shape=(None, 1))
    x = x_in
    skips = []

    for i, d in enumerate(dilations):
        k = 1 if i == 0 else int(kernel)
        x = TCNBlock(
            ch_out=ch,
            kernel=k,
            dilation=int(d),
            causal=causal,
            ch_hidden=ch_hidden,
            use_layernorm=use_layernorm,
            weight_decay=weight_decay,
        )(x)

        if use_skips:
            reg = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
            skips.append(tf.keras.layers.Conv1D(ch, 1, padding="same", kernel_regularizer=reg)(x))

    if use_skips and len(skips) > 0:
        x = tf.keras.layers.Add()(skips) if len(skips) > 1 else skips[0]

    # only last timestep
    x = tf.keras.layers.Lambda(lambda t: t[:, -1:, :])(x)

    reg_out = tf.keras.regularizers.l2(float(weight_decay)) if weight_decay and weight_decay > 0.0 else None
    x = tf.keras.layers.Conv1D(1, 1, padding="same", kernel_regularizer=reg_out)(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(x_in, x, name="D")

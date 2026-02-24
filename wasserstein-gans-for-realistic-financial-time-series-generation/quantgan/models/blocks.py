"""Neural network building blocks."""

import tensorflow as tf


def l2_reg(weight_decay: float):
    """Return an L2 regularizer or None for weight_decay<=0.
    
    Args:
        weight_decay: L2 regularization strength
        
    Returns:
        L2 regularizer or None
    """
    wd = float(weight_decay) if weight_decay is not None else 0.0
    return tf.keras.regularizers.l2(wd) if wd > 0.0 else None


def conv1d(filters, kernel, dilation, causal=True, weight_decay=0.0):
    """Create a 1D convolutional layer.
    
    Args:
        filters: Number of filters
        kernel: Kernel size
        dilation: Dilation rate
        causal: Use causal padding
        weight_decay: L2 regularization strength
        
    Returns:
        Conv1D layer
    """
    padding = "causal" if causal else "same"
    reg = l2_reg(weight_decay)
    return tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel,
        dilation_rate=dilation,
        padding=padding,
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
    )


def aggregate_skip_connections(skips):
    """Aggregate skip connections using addition.
    
    Args:
        skips: List of skip connection tensors
        
    Returns:
        Aggregated tensor or None if empty
    """
    if not skips or len(skips) == 0:
        return None
    return tf.keras.layers.Add()(skips) if len(skips) > 1 else skips[0]


@tf.keras.utils.register_keras_serializable(package="quantgan")
class TCNBlock(tf.keras.layers.Layer):
    """Temporal Convolutional Network block.
    
    Architecture: PReLU(Conv(PReLU(Conv(x))))
    Optional: LayerNorm after Conv (before PReLU).
    """

    def __init__(
        self,
        ch_out,
        kernel,
        dilation,
        causal=True,
        ch_hidden=None,
        use_layernorm=False,
        weight_decay=0.0,
    ):
        """Initialize TCN block.
        
        Args:
            ch_out: Output channels
            kernel: Kernel size
            dilation: Dilation rate
            causal: Use causal convolutions
            ch_hidden: Hidden channels (default: same as ch_out)
            use_layernorm: Apply layer normalization
            weight_decay: L2 regularization strength
        """
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.causal = causal
        self.weight_decay = weight_decay if weight_decay is not None else 0.0
        self.ch_out = ch_out
        self.ch_hidden = ch_hidden if ch_hidden is not None else ch_out
        self.use_layernorm = use_layernorm

        self.conv1 = conv1d(
            self.ch_hidden, self.kernel, self.dilation, self.causal,
            weight_decay=self.weight_decay
        )
        self.ln1 = (
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-3, scale=True)
            if self.use_layernorm
            else None
        )
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1])

        self.conv2 = conv1d(
            self.ch_out, self.kernel, self.dilation, self.causal,
            weight_decay=self.weight_decay
        )
        self.ln2 = (
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-3, scale=True)
            if self.use_layernorm
            else None
        )
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1])

    def call(self, x, training=False):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, channels)
            training: Training mode
            
        Returns:
            Output tensor
        """
        y = self.conv1(x)
        if self.ln1 is not None:
            y = self.ln1(y)
        y = self.prelu1(y)

        y = self.conv2(y)
        if self.ln2 is not None:
            y = self.ln2(y)
        y = self.prelu2(y)
        return y

    def get_config(self):
        """Get layer configuration."""
        cfg = super().get_config()
        cfg.update({
            "ch_out": self.ch_out,
            "kernel": self.kernel,
            "dilation": self.dilation,
            "causal": self.causal,
            "ch_hidden": self.ch_hidden,
            "use_layernorm": self.use_layernorm,
            "weight_decay": self.weight_decay,
        })
        return cfg

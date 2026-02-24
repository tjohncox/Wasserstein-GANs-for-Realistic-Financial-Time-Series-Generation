"""Models module for QuantGAN."""

from quantgan.models.blocks import TCNBlock, l2_reg, conv1d
from quantgan.models.generator import build_G_svnn, build_G_pure_tcn
from quantgan.models.discriminator import build_D
from quantgan.models.registry import build_generator, build_discriminator

__all__ = [
    "TCNBlock",
    "l2_reg",
    "conv1d",
    "build_G_svnn",
    "build_G_pure_tcn",
    "build_D",
    "build_generator",
    "build_discriminator",
]

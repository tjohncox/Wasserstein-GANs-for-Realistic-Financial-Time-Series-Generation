"""Training module for QuantGAN."""

from quantgan.training.schedule import EpochDecay
from quantgan.training.trainer import WGANGPTrainer

__all__ = [
    "EpochDecay",
    "WGANGPTrainer",
]

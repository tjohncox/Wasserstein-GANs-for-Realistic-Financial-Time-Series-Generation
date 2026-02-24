"""Learning rate schedules for training."""

import tensorflow as tf


class EpochDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with epoch-based decay.
    
    Decays learning rate by decay_rate every epoch after decay_start.
    """

    def __init__(self, lr0, steps_per_epoch_effective, decay_start, decay_rate, min_lr):
        """Initialize schedule.
        
        Args:
            lr0: Initial learning rate
            steps_per_epoch_effective: Steps per epoch
            decay_start: Epoch to start decay
            decay_rate: Decay factor per epoch
            min_lr: Minimum learning rate
        """
        super().__init__()
        self.lr0 = float(lr0)
        self.spe = float(steps_per_epoch_effective)
        self.decay_start = float(decay_start)
        self.decay_rate = float(decay_rate)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        """Compute learning rate at given step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate
        """
        step = tf.cast(step, tf.float32)
        epoch = tf.floor(step / self.spe)
        k = tf.maximum(epoch - self.decay_start, 0.0)
        lr = self.lr0 * tf.pow(self.decay_rate, k)
        return tf.maximum(lr, self.min_lr)

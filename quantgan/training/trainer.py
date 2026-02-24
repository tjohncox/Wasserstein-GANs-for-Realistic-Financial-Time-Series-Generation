"""WGAN-GP trainer for QuantGAN."""

import os
import logging
import numpy as np
import tensorflow as tf
from scipy.stats import kurtosis

from quantgan.models import build_generator, build_discriminator
from quantgan.evaluation.metrics import acf_tf, leverage_tf, tf_kurtosis_per_batch
from quantgan.training.schedule import EpochDecay
from quantgan.utils.io import write_weights_meta

logger = logging.getLogger(__name__)


class WGANGPTrainer:
    """WGAN-GP trainer for financial time series generation."""

    def __init__(
        self,
        model_cfg,
        train_cfg,
        debug_cfg,
        window_len,
        steps_per_epoch,
        vol_lags,
        lev_lags,
        acf_abs_real_tgt,
        acf_sq_real_tgt,
        lev_real_tgt,
    ):
        """Initialize trainer.
        
        Args:
            model_cfg: ModelConfig instance
            train_cfg: TrainConfig instance
            debug_cfg: DebugConfig instance
            window_len: Window length for training
            steps_per_epoch: Steps per epoch
            vol_lags: Lags for volatility ACF
            lev_lags: Lags for leverage
            acf_abs_real_tgt: Target ACF for absolute returns
            acf_sq_real_tgt: Target ACF for squared returns
            lev_real_tgt: Target leverage correlation
        """
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.dbg = debug_cfg

        # Store dimensions (no unnecessary int() casting if already ints)
        self.window_len = window_len
        self.burn_in = window_len - 1
        self.steps_per_epoch = steps_per_epoch
        self.vol_lags = vol_lags
        self.lev_lags = lev_lags

        self.acf_abs_real_tgt = acf_abs_real_tgt
        self.acf_sq_real_tgt = acf_sq_real_tgt
        self.lev_real_tgt = lev_real_tgt

        # Build models
        self.netG = build_generator(self.mcfg)
        self.netD = build_discriminator(self.mcfg)

        # Learning rate schedules
        lr_generator = EpochDecay(
            self.tcfg.lr_g0,
            self.steps_per_epoch,
            self.tcfg.decay_start,
            self.tcfg.decay_rate,
            self.tcfg.min_lr,
        )
        lr_discriminator = EpochDecay(
            self.tcfg.lr_d0,
            self.steps_per_epoch * self.tcfg.n_critic,
            self.tcfg.decay_start,
            self.tcfg.decay_rate,
            self.tcfg.min_lr,
        )

        # Optimizers
        self.opt_generator = tf.keras.optimizers.Adam(
            learning_rate=lr_generator, beta_1=0.1, beta_2=0.999
        )
        self.opt_discriminator = tf.keras.optimizers.Adam(
            learning_rate=lr_discriminator, beta_1=0.1, beta_2=0.999
        )

        # Build models once
        dummy_z = tf.random.normal([2, self.window_len + self.burn_in, self.mcfg.z_dim])
        _ = self.netG(dummy_z, training=False)

    def gradient_penalty(self, real, fake):
        """Compute gradient penalty for WGAN-GP.
        
        Args:
            real: Real data batch
            fake: Fake data batch
            
        Returns:
            Tuple of (gp, gp_norm_mean, gp_norm_max)
        """
        batch_size = tf.shape(real)[0]
        eps = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        x_hat = eps * real + (1.0 - eps) * fake

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(x_hat)
            d_hat = self.netD(x_hat, training=True)
            d_hat_sum = tf.reduce_sum(d_hat)

        grads = gp_tape.gradient(d_hat_sum, x_hat)
        grads = tf.reshape(grads, [batch_size, -1])
        grad_norm = tf.norm(grads, axis=1)
        gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return gp, tf.reduce_mean(grad_norm), tf.reduce_max(grad_norm)

    @tf.function
    def d_train_step(self, real):
        """Single discriminator training step.
        
        Args:
            real: Real data batch
            
        Returns:
            Tuple of training metrics
        """
        z_len = tf.shape(real)[1] + self.burn_in
        z = tf.random.normal([tf.shape(real)[0], z_len, self.mcfg.z_dim])

        fake_full = self.netG(z, training=False)
        fake = fake_full[:, self.burn_in:, :]
        fake = fake[:, :tf.shape(real)[1], :]
        fake_det = tf.stop_gradient(fake)

        d_vars = self.netD.trainable_variables
        with tf.GradientTape() as tape:
            d_real = self.netD(real, training=True)
            d_fake = self.netD(fake_det, training=True)

            loss_wasserstein = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            gp, gp_norm_mean, gp_norm_max = self.gradient_penalty(real, fake_det)
            loss_discriminator = loss_wasserstein + self.tcfg.lambda_gp * gp

        grads = tape.gradient(loss_discriminator, d_vars)
        self.opt_discriminator.apply_gradients(zip(grads, d_vars))
        return loss_discriminator, loss_wasserstein, gp, gp_norm_mean, gp_norm_max, d_real, d_fake, fake_det

    @tf.function
    def g_train_step(self, z, real_batch):
        """Single generator training step.
        
        Args:
            z: Latent noise
            real_batch: Real data batch (for shape reference)
            
        Returns:
            Tuple of (loss_generator, fake_det)
        """
        g_vars = self.netG.trainable_variables
        with tf.GradientTape() as tape:
            fake_full = self.netG(z, training=True)
            fake = fake_full[:, self.burn_in:, :]
            fake = fake[:, :tf.shape(real_batch)[1], :]
            d_fake = self.netD(fake, training=False)
            loss_generator = -tf.reduce_mean(d_fake)

        grads = tape.gradient(loss_generator, g_vars)
        self.opt_generator.apply_gradients(zip(grads, g_vars))

        fake_det = tf.stop_gradient(fake)
        return loss_generator, fake_det

    def monitor_metrics(self, real_batch, fake_det):
        """Compute monitoring metrics.
        
        These are used for logging and not part of the optimization objective.
        
        Args:
            real_batch: Real data batch
            fake_det: Fake data batch (detached)
            
        Returns:
            Tuple of (moments_loss, drift_loss, acf_abs_loss, acf_sq_loss, leverage_loss)
        """
        mean_real = tf.reduce_mean(real_batch)
        std_real = tf.math.reduce_std(real_batch)
        kurt_real = tf_kurtosis_per_batch(real_batch)

        mean_fake = tf.reduce_mean(fake_det)
        std_fake = tf.math.reduce_std(fake_det)
        kurt_fake = tf_kurtosis_per_batch(fake_det)

        moments_loss = (
            tf.square(mean_fake - mean_real)
            + tf.square(std_fake - std_real)
            + tf.square(kurt_fake - kurt_real)
        )
        drift_loss = tf.square(mean_fake - mean_real)

        acf_abs_fake = tf.reduce_mean(
            acf_tf(tf.abs(fake_det), self.vol_lags), axis=0
        )
        acf_sq_fake = tf.reduce_mean(
            acf_tf(tf.square(fake_det), self.vol_lags), axis=0
        )
        acf_abs_loss = tf.reduce_mean(tf.square(acf_abs_fake - self.acf_abs_real_tgt))
        acf_sq_loss = tf.reduce_mean(tf.square(acf_sq_fake - self.acf_sq_real_tgt))

        leverage_fake = tf.reduce_mean(leverage_tf(fake_det, self.lev_lags), axis=0)
        leverage_loss = tf.reduce_mean(tf.square(leverage_fake - self.lev_real_tgt))

        return moments_loss, drift_loss, acf_abs_loss, acf_sq_loss, leverage_loss

    def _generate_runs_raw(self, evaluator, T_target, n_runs):
        """Generate raw paths using evaluator.
        
        Args:
            evaluator: PaperEvaluator instance
            T_target: Target length
            n_runs: Number of runs to generate
            
        Returns:
            Array of generated paths (n_runs, T_target)
        """
        fake_runs = evaluator.sample_paths_raw(
            self.netG,
            z_dim=self.mcfg.z_dim,
            M=n_runs,
            Ttilde=T_target,
            burn_in=self.burn_in,
        )
        return fake_runs

    def _basic_checks(self, real_logret, fake_runs):
        """Perform basic sanity checks on generated data.
        
        Args:
            real_logret: Real log returns
            fake_runs: Generated runs (M, T)
        """
        fake_flat = fake_runs.reshape(-1)
        logger.info("[FINAL CHECK] Real mean=%.6g, Fake mean=%.6g", 
                   real_logret.mean(), fake_flat.mean())
        logger.info("[FINAL CHECK] Real std=%.6g, Fake std=%.6g", 
                   real_logret.std(), fake_flat.std())
        
        real_kurt = kurtosis(real_logret, fisher=False, bias=False)
        fake_kurt = kurtosis(fake_flat, fisher=False, bias=False)
        logger.info("[FINAL CHECK] Real kurt=%.3f, Fake kurt=%.3f", real_kurt, fake_kurt)
        
        maxabs_fake = np.max(np.abs(fake_flat))
        q999_fake = np.quantile(np.abs(fake_flat), 0.999)
        logger.info("[FINAL CHECK] Fake maxabs=%.4g q99.9=%.4g", maxabs_fake, q999_fake)

    def train(
        self,
        train_ds,
        evaluator,
        real_logret,
        out_dir,
        run_seed,
        gen_type,
        n_plot_runs=50,
    ):
        """Train the WGAN-GP model.
        
        Args:
            train_ds: Training dataset (TensorFlow dataset)
            evaluator: PaperEvaluator instance
            real_logret: Real log returns for evaluation
            out_dir: Output directory for saving weights
            run_seed: Random seed for this run
            gen_type: Generator type string
            n_plot_runs: Number of runs to generate for final checks
            
        Returns:
            Dictionary with training results
        """
        logger.info("[TRAIN] Starting training for %d epochs", self.tcfg.epochs)
        logger.info("[MODEL] Generator: %d params, Discriminator: %d params",
            self.netG.count_params(),
            len(self.netD.trainable_variables),
        )

        best_score = np.inf
        best_weights = None
        best_epoch = -1
        best_parts = None

        for epoch in range(self.tcfg.epochs):
            last_discriminator_loss = 0.0
            last_generator_loss = 0.0
            last_wasserstein_loss = 0.0
            last_gradient_penalty = 0.0

            for batch_idx, real_batch in enumerate(train_ds.take(self.steps_per_epoch)):

                # Pretrain discriminator
                if epoch < self.tcfg.pretrain_d_epochs:
                    for _ in range(self.tcfg.n_critic):
                        (
                            loss_discriminator,
                            loss_wasserstein,
                            gp,
                            gp_norm_mean,
                            gp_norm_max,
                            *_,
                        ) = self.d_train_step(real_batch)
                    if batch_idx == 0 and self.dbg.verbose:
                        logger.info(
                            "[PRETRAIN D] Epoch %02d | loss_D=%.4g W=%.4g GP=%.4g | "
                            "gp_norm_mean=%.3f gp_norm_max=%.3f",
                            epoch,
                            float(loss_discriminator.numpy()),
                            float(loss_wasserstein.numpy()),
                            float(gp.numpy()),
                            float(gp_norm_mean.numpy()),
                            float(gp_norm_max.numpy()),
                        )
                    continue

                # Train discriminator
                for _ in range(self.tcfg.n_critic):
                    (
                        loss_discriminator,
                        loss_wasserstein,
                        gp,
                        gp_norm_mean,
                        gp_norm_max,
                        d_real,
                        d_fake,
                        fake_det,
                    ) = self.d_train_step(real_batch)

                # Train generator
                z = tf.random.normal(
                    [
                        tf.shape(real_batch)[0],
                        self.window_len + self.burn_in,
                        self.mcfg.z_dim,
                    ]
                )
                loss_generator, fake_det_g = self.g_train_step(z, real_batch)

                last_discriminator_loss = float(loss_discriminator.numpy())
                last_wasserstein_loss = float(loss_wasserstein.numpy())
                last_gradient_penalty = float(gp.numpy())
                last_generator_loss = float(loss_generator.numpy())

                # Logging at first batch
                if batch_idx == 0 and self.dbg.verbose:
                    moments_loss, drift_loss, acf_abs_loss, acf_sq_loss, leverage_loss = self.monitor_metrics(
                        real_batch, fake_det_g
                    )
                    logger.info(
                        "[Epoch %02d Batch %04d] loss_D=%.4g (W=%.4g, GP=%.4g) | loss_G=%.4g | "
                        "gp_norm_mean=%.3f gp_norm_max=%.3f | moments=%.4g drift=%.4g "
                        "acf|r|=%.4g acf_r²=%.4g lev=%.4g",
                        epoch,
                        batch_idx,
                        last_discriminator_loss,
                        last_wasserstein_loss,
                        last_gradient_penalty,
                        last_generator_loss,
                        float(gp_norm_mean.numpy()),
                        float(gp_norm_max.numpy()),
                        float(moments_loss.numpy()),
                        float(drift_loss.numpy()),
                        float(acf_abs_loss.numpy()),
                        float(acf_sq_loss.numpy()),
                        float(leverage_loss.numpy()),
                    )
                    logger.info(
                        "  D(real) mean: %.4g | D(fake) mean: %.4g",
                        float(tf.reduce_mean(d_real).numpy()),
                        float(tf.reduce_mean(d_fake).numpy()),
                    )

                    if self.dbg.debug_tails:
                        fake_train = fake_det_g.numpy().reshape(-1)
                        logger.info(
                            "  [DEBUG FAKE TRAIN] maxabs=%.4g q99.9=%.4g q99.99=%.4g",
                            float(np.max(np.abs(fake_train))),
                            float(np.quantile(np.abs(fake_train), 0.999)),
                            float(np.quantile(np.abs(fake_train), 0.9999)),
                        )

                        if (epoch % self.dbg.debug_raw_every) == 0:
                            fake_seq = fake_det_g.numpy()[0, :, 0]
                            raw_seq = evaluator.preproc.inverse_transform(fake_seq)
                            logger.info(
                                "  [DEBUG FAKE RAW] maxabs=%.4g q01,q05,q95,q99=%s",
                                float(np.max(np.abs(raw_seq))),
                                np.quantile(raw_seq, [0.01, 0.05, 0.95, 0.99]),
                            )

            # Evaluation after each epoch
            raw_stats = evaluator.raw_stats(
                self.netG, z_dim=self.mcfg.z_dim, T_eval=800, burn_in=self.burn_in,
                n_runs=3
            )
            mean_err = abs(raw_stats["mean"] - evaluator.real_mean) / evaluator.real_std
            std_err = abs(raw_stats["std"] - evaluator.real_std) / evaluator.real_std
            q_err = float(
                np.mean(np.abs(raw_stats["qs"] - evaluator.real_qs)) / evaluator.real_std
            )

            paper_score = np.nan
            parts = None

            if (epoch % self.tcfg.sel_every) == 0 or (epoch == self.tcfg.epochs - 1):
                fake_sel = evaluator.sample_paths_raw(
                    self.netG,
                    z_dim=self.mcfg.z_dim,
                    M=self.tcfg.sel_m,
                    Ttilde=self.tcfg.sel_ttilde,
                    burn_in=self.burn_in,
                )
                paper_score, parts = evaluator.paper_score(fake_sel)

                if paper_score < best_score:
                    best_score = float(paper_score)
                    best_epoch = epoch
                    best_parts = parts
                    best_weights = self.netG.get_weights()
                    logger.info(
                        "[Epoch %02d] → Saved BEST generator (paper_score=%.6g)",
                        epoch,
                        best_score,
                    )

            # Epoch summary logging
            if self.dbg.verbose:
                if parts is None:
                    logger.info(
                        "[Epoch %02d] Complete | loss_D=%.4g loss_G=%.4g (no paper-eval this epoch) | "
                        "raw_mean=%.3g raw_std=%.3g mean_err=%.3g std_err=%.3g q_err=%.3g",
                        epoch,
                        last_discriminator_loss,
                        last_generator_loss,
                        raw_stats["mean"],
                        raw_stats["std"],
                        mean_err,
                        std_err,
                        q_err,
                    )
                else:
                    dy = parts["dy_by_t"]
                    logger.info(
                        "[Epoch %02d] Complete | loss_D=%.4g loss_G=%.4g | PAPER_score=%.6g "
                        "acf_abs=%.3g acf_x=%.3g acf_sq=%.3g lev=%.3g dy_sum=%.3g "
                        "DY1=%.3g DY5=%.3g DY20=%.3g DY100=%.3g | "
                        "raw_mean=%.3g raw_std=%.3g mean_err=%.3g std_err=%.3g q_err=%.3g",
                        epoch,
                        last_discriminator_loss,
                        last_generator_loss,
                        paper_score,
                        parts["acf_abs"],
                        parts["acf_x"],
                        parts["acf_sq"],
                        parts["lev"],
                        parts["dy_sum"],
                        dy.get(1, np.nan),
                        dy.get(5, np.nan),
                        dy.get(20, np.nan),
                        dy.get(100, np.nan),
                        raw_stats["mean"],
                        raw_stats["std"],
                        mean_err,
                        std_err,
                        q_err,
                    )

        # Restore best weights
        if best_weights is not None:
            self.netG.set_weights(best_weights)

        logger.info("\n=== BEST PAPER MODEL ===")
        logger.info("best_epoch=%d | best_paper_score=%.6g", best_epoch, best_score)
        if best_parts is not None:
            logger.info("best_parts: %s", best_parts)

        # Save weights
        os.makedirs(out_dir, exist_ok=True)
        weights_path = os.path.join(
            out_dir, f"bestG_{gen_type}_seed{run_seed}_epoch{best_epoch}.weights.h5"
        )
        self.netG.save_weights(weights_path)
        write_weights_meta(weights_path, self.mcfg)

        # Generate final runs for checks
        fake_runs = self._generate_runs_raw(
            evaluator, T_target=len(real_logret), n_runs=n_plot_runs
        )
        self._basic_checks(real_logret, fake_runs)

        return {
            "seed": run_seed,
            "gen_type": gen_type,
            "best_score": float(best_score),
            "best_epoch": best_epoch,
            "best_weights_path": weights_path,
        }

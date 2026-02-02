# train.py
import os
import random
import numpy as np
import yfinance as yf
import tensorflow as tf
from scipy.stats import kurtosis

from config import DataConfig, PreprocessConfig, DatasetConfig, ModelConfig, TrainConfig, DebugConfig, RunConfig
from preprocess import LambertWPreprocessor
from models import build_G_svnn, build_G_pure_tcn, build_D
from metrics import acf_tf, leverage_tf, tf_kurtosis_per_batch, PaperEvaluator, Plotter


# ============================================================
# Utils
# ============================================================
def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Data
# ============================================================
class YFinanceSource:
    def __init__(self, cfg):
        self.cfg = cfg

    def fetch(self):
        tk = yf.Ticker(self.cfg.ticker)
        df = tk.history(
            start=self.cfg.start,
            end=self.cfg.end,
            interval=self.cfg.interval,
            auto_adjust=False,
            actions=True,
        )
        if df is None or len(df) == 0:
            raise ValueError("[YF] Download returned empty dataframe. Check network / rate limit / ticker.")
        if "Close" not in df.columns:
            raise ValueError(f"[YF] 'Close' not in columns: {list(df.columns)}")
        return df

    @staticmethod
    def log_returns_from_close(df):
        close = df["Close"].dropna()
        logret = np.log(close / close.shift(1)).dropna().values.astype(np.float64)
        if len(logret) <= 0:
            raise ValueError("[DATA] logret is empty.")
        return logret


# ============================================================
# Dataset
# ============================================================
def make_windows_np(x, win):
    x = np.asarray(x, dtype=np.float32)
    win = int(win)
    if len(x) < win:
        raise ValueError(f"Not enough data for windows: len(x)={len(x)} win={win}")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        X = sliding_window_view(x, window_shape=win)  # (N, win)
        return np.asarray(X, dtype=np.float32)
    except Exception:
        N = len(x) - win + 1
        return np.stack([x[i:i + win] for i in range(N)], axis=0).astype(np.float32)


def window_sampling_probs(series_len, win_len):
    series_len = int(series_len)
    win_len = int(win_len)
    Nw = series_len - win_len + 1

    count = np.zeros(series_len, dtype=np.int64)
    for t in range(series_len):
        s_min = max(0, t - win_len + 1)
        s_max = min(t, Nw - 1)
        count[t] = max(0, s_max - s_min + 1)

    inv = 1.0 / np.maximum(count, 1)
    w = np.zeros(Nw, dtype=np.float64)
    for s in range(Nw):
        w[s] = float(inv[s:s + win_len].sum())

    w = np.maximum(w, 1e-12)
    p = w / w.sum()
    return p, count


class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, r_train):
        win = int(self.cfg.window_len)
        B = int(self.cfg.batch_size)

        X = make_windows_np(r_train, win)[:, :, None]  # (N, win, 1)
        Nw = int(X.shape[0])
        steps_per_epoch = max(1, Nw // B)

        if self.cfg.weighted_sampling:
            P_WIN, _ = window_sampling_probs(len(r_train), win)
            if B > Nw:
                raise ValueError(f"BATCH({B}) > Nw({Nw}). Reduce batch_size or window_len.")

            rng = np.random.default_rng(self.cfg.seed)

            def gen():
                while True:
                    idx = rng.choice(Nw, size=B, replace=False, p=P_WIN)
                    yield X[idx]

            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=tf.TensorSpec(shape=(B, win, 1), dtype=tf.float32)
            ).prefetch(tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices(X)
            ds = ds.shuffle(Nw, seed=self.cfg.seed, reshuffle_each_iteration=True)
            ds = ds.batch(B, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        return ds, X, steps_per_epoch


# ============================================================
# Optim schedule
# ============================================================
class EpochDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr0, steps_per_epoch_effective, decay_start, decay_rate, min_lr):
        super().__init__()
        self.lr0 = float(lr0)
        self.spe = float(steps_per_epoch_effective)
        self.decay_start = float(decay_start)
        self.decay_rate = float(decay_rate)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        epoch = tf.floor(step / self.spe)
        k = tf.maximum(epoch - self.decay_start, 0.0)
        lr = self.lr0 * tf.pow(self.decay_rate, k)
        return tf.maximum(lr, self.min_lr)


# ============================================================
# Model build / load
# ============================================================
def build_generator_from_cfg(model_cfg):
    gen_type = str(getattr(model_cfg, "generator_type", "svnn")).lower().strip()

    if gen_type == "svnn":
        return build_G_svnn(
            z_dim=model_cfg.z_dim,
            ch=model_cfg.g_ch,
            ch_hidden=model_cfg.g_ch_hidden,
            kernel=model_cfg.kernel,
            dilations=model_cfg.dilations,
            causal=model_cfg.causal,
            use_skips=model_cfg.use_skip_connections,
            use_soft_clip=model_cfg.g_use_soft_clip,
            soft_clip_c=model_cfg.g_soft_clip_c,
            constrained_innovation=model_cfg.g_constrained_innovation,
            eps_hidden=model_cfg.g_eps_hidden,
            use_layernorm=model_cfg.g_use_layernorm,
            weight_decay=model_cfg.g_weight_decay,
            sigma_mode=model_cfg.g_sigma_mode,
        )

    if gen_type == "pure_tcn":
        return build_G_pure_tcn(
            z_dim=model_cfg.z_dim,
            ch=model_cfg.g_ch,
            ch_hidden=model_cfg.g_ch_hidden,
            kernel=model_cfg.kernel,
            dilations=model_cfg.dilations,
            causal=model_cfg.causal,
            use_skips=model_cfg.use_skip_connections,
            use_soft_clip=model_cfg.g_use_soft_clip,
            soft_clip_c=model_cfg.g_soft_clip_c,
            use_layernorm=model_cfg.g_use_layernorm,
            weight_decay=model_cfg.g_weight_decay,
        )

    raise ValueError(f"Unknown generator_type: {model_cfg.generator_type}. Use 'svnn' or 'pure_tcn'.")


def build_and_load_generator(model_cfg, window_len, weights_path, seed=0):
    netG = build_generator_from_cfg(model_cfg)

    burn_in = int(window_len - 1)
    _z = tf.random.normal([1, 10 + burn_in, int(model_cfg.z_dim)], seed=int(seed))
    _ = netG(_z, training=False)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    netG.load_weights(weights_path)
    print("[LOAD] weights loaded:", weights_path)
    return netG


def generate_M_paths_raw(netG, preproc, M, Ttilde, window_len, z_dim, batch=50, seed=0):
    burn_in = int(window_len - 1)
    M = int(M)
    Ttilde = int(Ttilde)
    z_dim = int(z_dim)
    batch = int(batch)

    g = tf.random.Generator.from_seed(int(seed))

    outs = np.zeros((M, Ttilde), dtype=np.float64)
    done = 0
    while done < M:
        b = min(batch, M - done)
        z = g.normal([b, Ttilde + burn_in, z_dim])
        fake_full = netG(z, training=False).numpy()[..., 0]  # (b, Ttilde+burn_in)
        fake_used = fake_full[:, burn_in:burn_in + Ttilde]
        outs[done:done + b] = preproc.inverse_transform(fake_used)
        done += b

    return outs


# ============================================================
# Trainer
# ============================================================
class WGANGPTrainer:
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
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.dbg = debug_cfg

        self.window_len = int(window_len)
        self.burn_in = int(window_len - 1)
        self.steps_per_epoch = int(steps_per_epoch)

        self.vol_lags = int(vol_lags)
        self.lev_lags = int(lev_lags)

        self.acf_abs_real_tgt = acf_abs_real_tgt
        self.acf_sq_real_tgt = acf_sq_real_tgt
        self.lev_real_tgt = lev_real_tgt

        # --- Build G (based on generator_type)
        self.netG = build_generator_from_cfg(self.mcfg)

        # --- Build D
        self.netD = build_D(
            ch=self.mcfg.d_ch,
            ch_hidden=self.mcfg.d_ch_hidden,
            kernel=self.mcfg.kernel,
            dilations=self.mcfg.dilations,
            causal=self.mcfg.causal,
            use_skips=self.mcfg.use_skip_connections,
            use_layernorm=self.mcfg.d_use_layernorm,
            weight_decay=self.mcfg.d_weight_decay,
        )

        lrG = EpochDecay(self.tcfg.lr_g0, self.steps_per_epoch, self.tcfg.decay_start, self.tcfg.decay_rate, self.tcfg.min_lr)
        lrD = EpochDecay(self.tcfg.lr_d0, self.steps_per_epoch * self.tcfg.n_critic, self.tcfg.decay_start, self.tcfg.decay_rate, self.tcfg.min_lr)
        self.optG = tf.keras.optimizers.Adam(learning_rate=lrG, beta_1=0.1, beta_2=0.999)
        self.optD = tf.keras.optimizers.Adam(learning_rate=lrD, beta_1=0.1, beta_2=0.999)

        # build once
        _z = tf.random.normal([2, self.window_len + self.burn_in, self.mcfg.z_dim])
        _ = self.netG(_z, training=False)

    def gradient_penalty(self, real, fake):
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

            loss_w = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            gp, gp_norm_mean, gp_norm_max = self.gradient_penalty(real, fake_det)
            lossD = loss_w + self.tcfg.lambda_gp * gp

        grads = tape.gradient(lossD, d_vars)
        self.optD.apply_gradients(zip(grads, d_vars))
        return lossD, loss_w, gp, gp_norm_mean, gp_norm_max, d_real, d_fake, fake_det

    @tf.function
    def g_train_step(self, z, real_batch):
        g_vars = self.netG.trainable_variables
        with tf.GradientTape() as tape:
            fake_full = self.netG(z, training=True)
            fake = fake_full[:, self.burn_in:, :]
            fake = fake[:, :tf.shape(real_batch)[1], :]
            d_fake = self.netD(fake, training=False)
            lossG = -tf.reduce_mean(d_fake)

        grads = tape.gradient(lossG, g_vars)
        self.optG.apply_gradients(zip(grads, g_vars))

        fake_det = tf.stop_gradient(fake)

        m_real = tf.reduce_mean(real_batch)
        s_real = tf.math.reduce_std(real_batch)
        k_real = tf_kurtosis_per_batch(real_batch)

        m_fake = tf.reduce_mean(fake_det)
        s_fake = tf.math.reduce_std(fake_det)
        k_fake = tf_kurtosis_per_batch(fake_det)

        mom = tf.square(m_fake - m_real) + tf.square(s_fake - s_real) + tf.square(k_fake - k_real)
        drift = tf.square(m_fake - m_real)

        acf_abs_fake = tf.reduce_mean(acf_tf(tf.abs(fake_det), self.vol_lags), axis=0)
        acf_sq_fake = tf.reduce_mean(acf_tf(tf.square(fake_det), self.vol_lags), axis=0)
        l_abs = tf.reduce_mean(tf.square(acf_abs_fake - self.acf_abs_real_tgt))
        l_sq = tf.reduce_mean(tf.square(acf_sq_fake - self.acf_sq_real_tgt))

        lev_fake = tf.reduce_mean(leverage_tf(fake_det, self.lev_lags), axis=0)
        l_lev = tf.reduce_mean(tf.square(lev_fake - self.lev_real_tgt))

        return lossG, mom, drift, l_abs, l_sq, l_lev, fake_det

    def _generate_runs_raw(self, evaluator, T_target, n_runs):
        outs = []
        for _ in range(int(n_runs)):
            z = tf.random.normal([1, int(T_target) + self.burn_in, self.mcfg.z_dim])
            g_train = self.netG(z, training=False).numpy().reshape(-1)
            g_train = g_train[self.burn_in:]
            outs.append(evaluator.preproc.inverse_transform(g_train))
        return np.stack(outs, axis=0)

    @staticmethod
    def _basic_checks(real, fake_runs):
        fake_pool = fake_runs.reshape(-1)
        print("\n=== BASIC CHECKS ===")
        print("real:", real.shape, "fake_runs:", fake_runs.shape)
        print("mean(real)=", float(real.mean()), "std(real)=", float(real.std()))
        print("mean(fake_pool)=", float(fake_pool.mean()), "std(fake_pool)=", float(fake_pool.std()))
        print("[1-day] kurt_real=", float(kurtosis(real, fisher=False, bias=False)),
              "kurt_fake=", float(kurtosis(fake_pool, fisher=False, bias=False)))

    def fit(
        self,
        train_ds,
        evaluator,
        out_dir,
        real_logret,
        close_series,
        n_plot_runs,
        n_plot_paths,
        show_plots,
        save_plots,
        run_seed,
    ):
        gen_type = str(getattr(self.mcfg, "generator_type", "svnn"))

        print("[MODEL] gen_type=", gen_type,
              "CAUSAL=", self.mcfg.causal,
              "SKIPS=", self.mcfg.use_skip_connections,
              "T_D=", self.window_len,
              "Z_DIM=", self.mcfg.z_dim,
              "KERNEL=", self.mcfg.kernel,
              "DILATIONS=", list(self.mcfg.dilations),
              "G_vars=", len(self.netG.trainable_variables),
              "D_vars=", len(self.netD.trainable_variables))

        best_score = np.inf
        best_weights = None
        best_epoch = -1
        best_parts = None

        for e in range(int(self.tcfg.epochs)):
            lastD = lastG = lastW = lastGP = 0.0

            for b, real_batch in enumerate(train_ds.take(self.steps_per_epoch)):

                if e < int(self.tcfg.pretrain_d_epochs):
                    for _ in range(int(self.tcfg.n_critic)):
                        lossD, loss_w, gp, gp_norm_mean, gp_norm_max, *_ = self.d_train_step(real_batch)
                    if b == 0 and self.dbg.verbose:
                        print(f"[PRETRAIN D] E{e:02d} lossD={float(lossD.numpy()):.4g} "
                              f"W={float(loss_w.numpy()):.4g} GP={float(gp.numpy()):.4g} "
                              f"gp_norm_mean={float(gp_norm_mean.numpy()):.3f} gp_norm_max={float(gp_norm_max.numpy()):.3f}")
                    continue

                for _ in range(int(self.tcfg.n_critic)):
                    lossD, loss_w, gp, gp_norm_mean, gp_norm_max, d_real, d_fake, fake_det = self.d_train_step(real_batch)

                z = tf.random.normal([tf.shape(real_batch)[0], self.window_len + self.burn_in, self.mcfg.z_dim])
                lossG, mom, drift, l_abs, l_sq, l_lev, fake_det_g = self.g_train_step(z, real_batch)

                lastD = float(lossD.numpy())
                lastW = float(loss_w.numpy())
                lastGP = float(gp.numpy())
                lastG = float(lossG.numpy())

                if b == 0 and self.dbg.verbose:
                    print(f"[E{e:02d} B{b:04d}] "
                          f"lossD={lastD:.4g} (W={lastW:.4g}, GP={lastGP:.4g}) "
                          f"| lossG={lastG:.4g} "
                          f"| gp_norm_mean={float(gp_norm_mean.numpy()):.3f} gp_norm_max={float(gp_norm_max.numpy()):.3f} "
                          f"| mom={float(mom.numpy()):.4g} drift={float(drift.numpy()):.4g} "
                          f"acf|r|={float(l_abs.numpy()):.4g} acf r2={float(l_sq.numpy()):.4g} lev={float(l_lev.numpy()):.4g}")
                    print("  D(real) mean:", float(tf.reduce_mean(d_real).numpy()),
                          "D(fake) mean:", float(tf.reduce_mean(d_fake).numpy()))

                    if self.dbg.debug_tails:
                        ft = fake_det_g.numpy().reshape(-1)
                        print("  [DBG FAKE TRAIN] maxabs=", float(np.max(np.abs(ft))),
                              "q99.9=", float(np.quantile(np.abs(ft), 0.999)),
                              "q99.99=", float(np.quantile(np.abs(ft), 0.9999)))

                        if (e % int(self.dbg.debug_raw_every)) == 0:
                            ft_seq = fake_det_g.numpy()[0, :, 0]
                            raw_seq = evaluator.preproc.inverse_transform(ft_seq)
                            print("  [DBG FAKE RAW ] maxabs=", float(np.max(np.abs(raw_seq))),
                                  "q01,q05,q95,q99=", np.quantile(raw_seq, [0.01, 0.05, 0.95, 0.99]))

            raw = evaluator.raw_stats(self.netG, z_dim=self.mcfg.z_dim, T_eval=800, burn_in=self.burn_in, n_runs=3)
            mean_err = abs(raw["mean"] - evaluator.real_mean) / evaluator.real_std
            std_err = abs(raw["std"] - evaluator.real_std) / evaluator.real_std
            q_err = float(np.mean(np.abs(raw["qs"] - evaluator.real_qs)) / evaluator.real_std)

            paper_score = np.nan
            parts = None

            if (e % int(self.tcfg.sel_every)) == 0 or (e == int(self.tcfg.epochs) - 1):
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
                    best_epoch = int(e)
                    best_parts = parts
                    best_weights = self.netG.get_weights()
                    print(f"[E{e:02d}] -> saved BEST generator (paper_score={best_score:.6g})")

            if self.dbg.verbose:
                if parts is None:
                    print(f"[E{e:02d}] done lastD={lastD:.4g} lastG={lastG:.4g} "
                          f"(no paper-eval this epoch) | raw_mean={raw['mean']:.3g} raw_std={raw['std']:.3g} "
                          f"mean_err={mean_err:.3g} std_err={std_err:.3g} q_err={q_err:.3g}")
                else:
                    dy = parts["dy_by_t"]
                    print(f"[E{e:02d}] done lastD={lastD:.4g} lastG={lastG:.4g} "
                          f"PAPER_score={paper_score:.6g} "
                          f"acf_abs={parts['acf_abs']:.3g} acf_x={parts['acf_x']:.3g} acf_sq={parts['acf_sq']:.3g} lev={parts['lev']:.3g} "
                          f"dy_sum={parts['dy_sum']:.3g} "
                          f"DY1={dy.get(1, np.nan):.3g} DY5={dy.get(5, np.nan):.3g} DY20={dy.get(20, np.nan):.3g} DY100={dy.get(100, np.nan):.3g} "
                          f"| raw_mean={raw['mean']:.3g} raw_std={raw['std']:.3g} mean_err={mean_err:.3g} std_err={std_err:.3g} q_err={q_err:.3g}")

        if best_weights is not None:
            self.netG.set_weights(best_weights)

        print("\n=== BEST PAPER MODEL ===")
        print(f"best_epoch={best_epoch} best_paper_score={best_score:.6g}")
        if best_parts is not None:
            print("best_parts:", best_parts)

        os.makedirs(out_dir, exist_ok=True)
        weights_path = os.path.join(out_dir, f"bestG_{gen_type}_seed{run_seed}_epoch{best_epoch}.weights.h5")
        self.netG.save_weights(weights_path)

        fake_runs = self._generate_runs_raw(evaluator, T_target=len(real_logret), n_runs=n_plot_runs)
        self._basic_checks(real_logret, fake_runs)

        plotter = Plotter(show=show_plots, save=save_plots, out_dir=out_dir if save_plots else None)
        tag = f"_{gen_type}_seed{run_seed}_bestE{best_epoch}"

        plotter.plot_price_paths(
            close_series,
            fake_runs,
            n_paths=n_plot_paths,
            title=f"Price paths (Real + {min(int(n_plot_paths), int(fake_runs.shape[0]))} Fake){tag}",
            filename=f"price_paths{tag}.png" if save_plots else None,
        )
        plotter.plot_acf_bundle(real_logret, fake_runs, S=250, title_suffix=tag, filename_prefix=f"acf{tag}" if save_plots else None)
        plotter.plot_leverage(real_logret, fake_runs, S=250, title_suffix=tag, filename=f"leverage{tag}.png" if save_plots else None)

        return {
            "seed": int(run_seed),
            "gen_type": str(gen_type),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "best_weights_path": weights_path,
        }


# ============================================================
# Eval
# ============================================================
def eval_once(
    weights_path,
    out_dir,
    data_cfg,
    pre_cfg,
    ds_cfg,
    model_cfg,
    train_cfg,
    drift_align=True,
    do_plots=True,
    save_plots=False,
    M=500,
    Ttilde=4000,   # horizon for paper-metrics / long-run eval
    S_lags=250,
    seed=0,
    batch=50,
):
    """
    Eval pipeline:
      - load real data
      - fit preprocessor on real
      - build + load generator (svnn/pure_tcn via model_cfg.generator_type)
      - generate M fake paths of length Ttilde (raw log-returns)
      - optional drift-align (mean match)
      - compute paper_score on full length Ttilde
      - for plots that use real dates: slice fake paths to T_plot = len(real_series)

    This avoids Matplotlib "x and y must have same first dimension" when Ttilde != len(real_series).
    """
    set_all_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------
    # Load real data
    # ---------------------------
    src = YFinanceSource(data_cfg)
    df = src.fetch()
    close = df["Close"].dropna()
    real_series = src.log_returns_from_close(df).astype(np.float64)
    T_real = int(len(real_series))

    print("[EVAL DATA]", data_cfg.ticker, "T=", T_real,
          "mean=", float(real_series.mean()),
          "std=", float(real_series.std() + 1e-12),
          "kurt=", float(kurtosis(real_series, fisher=False, bias=False)))

    # ---------------------------
    # Rebuild preprocessing (fit on same real data window)
    # ---------------------------
    pre = LambertWPreprocessor(pre_cfg).fit(real_series)
    print("[EVAL PRE] state:", pre.summary())

    # ---------------------------
    # Build + load generator (respects model_cfg.generator_type)
    # ---------------------------
    gen_type = str(getattr(model_cfg, "generator_type", "svnn"))
    print("[EVAL MODEL] generator_type =", gen_type)

    netG = build_and_load_generator(
        model_cfg=model_cfg,
        window_len=ds_cfg.window_len,
        weights_path=weights_path,
        seed=seed,
    )

    # ---------------------------
    # Generate M paths in RAW space (length Ttilde)
    # ---------------------------
    Ttilde = int(Ttilde)
    fake_paths = generate_M_paths_raw(
        netG=netG,
        preproc=pre,
        M=int(M),
        Ttilde=Ttilde,
        window_len=ds_cfg.window_len,
        z_dim=model_cfg.z_dim,
        batch=int(batch),
        seed=seed,
    )

    print("[EVAL SIM raw] shape=", fake_paths.shape,
          "mean=", float(fake_paths.mean()),
          "std=", float(fake_paths.std() + 1e-12),
          "kurt=", float(kurtosis(fake_paths.reshape(-1), fisher=False, bias=False)))

    # ---------------------------
    # Optional drift alignment (mean-match) on FULL fake_paths
    # ---------------------------
    used_fake = fake_paths
    drift_shift = 0.0
    if drift_align:
        mu_target = float(real_series.mean())
        mu_fake = float(fake_paths.mean())
        drift_shift = mu_target - mu_fake
        used_fake = fake_paths + drift_shift
        print("[EVAL DRIFT] mu_target=", mu_target,
              "mu_fake=", mu_fake,
              "shift=", drift_shift,
              "aligned_mean=", float(used_fake.mean()))

    # ---------------------------
    # Metrics (PaperEvaluator) on FULL horizon (Ttilde)
    # ---------------------------
    evaluator = PaperEvaluator(
        real_series=real_series,
        preproc=pre,
        train_cfg=train_cfg,
        dy_base_t=len(real_series),
    )

    paper_score, parts = evaluator.paper_score(used_fake)

    print("\n[EVAL PAPER SCORE]")
    print("paper_score =", float(paper_score))
    print("parts =", parts)

    raw = evaluator.raw_stats(
        netG,
        z_dim=model_cfg.z_dim,
        T_eval=800,
        burn_in=ds_cfg.window_len - 1,
        n_runs=3,
    )
    mean_err = abs(raw["mean"] - evaluator.real_mean) / evaluator.real_std
    std_err = abs(raw["std"] - evaluator.real_std) / evaluator.real_std
    q_err = float(np.mean(np.abs(raw["qs"] - evaluator.real_qs)) / evaluator.real_std)

    print("\n[EVAL RAW STATS]")
    print("raw_mean=", raw["mean"], "raw_std=", raw["std"], "raw_qs=", raw["qs"])
    print("mean_err=", mean_err, "std_err=", std_err, "q_err=", q_err)

    # ---------------------------
    # Plots (slice to real horizon so dates and paths match)
    # ---------------------------
    used_fake_plot = used_fake
    if Ttilde != T_real:
        # slice to match real dates for plotting
        T_plot = min(T_real, Ttilde)
        used_fake_plot = used_fake[:, :T_plot]
        print(f"[EVAL PLOT] slicing fake paths for plots: Ttilde={Ttilde} -> T_plot={T_plot} (real T={T_real})")
    else:
        print(f"[EVAL PLOT] using full horizon for plots: T={T_real}")

    if do_plots:
        plotter = Plotter(show=True, save=bool(save_plots), out_dir=out_dir if save_plots else None)
        tag = os.path.splitext(os.path.basename(weights_path))[0]

        plotter.plot_price_paths(
            close_series=close,
            fake_logret_paths=used_fake_plot,
            n_paths=min(50, int(used_fake_plot.shape[0])),
            title=f"Price paths (Real + Fake) | {tag}",
            filename=f"price_paths_{tag}.png" if save_plots else None,
        )

        plotter.plot_acf_bundle(
            real_r=real_series[:used_fake_plot.shape[1]],
            fake_paths=used_fake_plot,
            S=int(S_lags),
            title_suffix=f" | {tag}",
            filename_prefix=f"acf_{tag}" if save_plots else None,
        )

        plotter.plot_leverage(
            real_r=real_series[:used_fake_plot.shape[1]],
            fake_paths=used_fake_plot,
            S=int(S_lags),
            title_suffix=f" | {tag}",
            filename=f"leverage_{tag}.png" if save_plots else None,
        )

    return {
        "weights_path": str(weights_path),
        "generator_type": str(getattr(model_cfg, "generator_type", "svnn")),
        "drift_align": bool(drift_align),
        "drift_shift": float(drift_shift),
        "paper_score": float(paper_score),
        "paper_parts": parts,
        "raw_stats": raw,
        "raw_err": {"mean_err": float(mean_err), "std_err": float(std_err), "q_err": float(q_err)},
        "fake_paths": used_fake,           # full horizon (Ttilde)
        "fake_paths_for_plots": used_fake_plot,  # sliced to match real horizon (or same as full)
        "T_real": int(T_real),
        "Ttilde": int(Ttilde),
    }



# ============================================================
# Main (ONE place to switch train/eval + generator)
# ============================================================
def main():
    data_cfg = DataConfig()
    pre_cfg = PreprocessConfig()
    ds_cfg = DatasetConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    dbg_cfg = DebugConfig()
    run_cfg = RunConfig()

    set_all_seeds(run_cfg.seed)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # ---------------------------
    # RUN SETTINGS (edit only here)
    # ---------------------------
    MODE = "eval"          # "train" or "eval"
    GENERATOR = "svnn"      # "svnn" or "pure_tcn"
    WEIGHTS_PATH = "./models/bestG_WGANGP_EMA_seed0_epoch225.weights.h5"  # used only if MODE == "eval"
    EPOCHS = 10             # used only if MODE == "train"
    SAVE_PLOTS = True       # used only if MODE == "eval" (and also affects training plots via run_cfg)

    model_cfg.generator_type = GENERATOR

    out_dir_train = os.path.join(os.getcwd(), run_cfg.out_dir)
    out_dir_eval = os.path.join(os.getcwd(), "quantgan_eval_outputs")

    # ---------------------------
    # EVAL
    # ---------------------------
    if MODE == "eval":
        if not WEIGHTS_PATH or not str(WEIGHTS_PATH).strip():
            raise ValueError("MODE='eval' but WEIGHTS_PATH is empty.")

        eval_once(
            weights_path=WEIGHTS_PATH,
            out_dir=out_dir_eval,
            data_cfg=data_cfg,
            pre_cfg=pre_cfg,
            ds_cfg=ds_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            drift_align=True,
            do_plots=True,
            save_plots=bool(SAVE_PLOTS),
            M=500,
            Ttilde=4000,
            S_lags=250,
            seed=run_cfg.seed,
            batch=50,
        )
        return

    # ---------------------------
    # TRAIN
    # ---------------------------
    if MODE != "train":
        raise ValueError("MODE must be 'train' or 'eval'.")

    train_cfg.epochs = int(EPOCHS)

    # data
    src = YFinanceSource(data_cfg)
    df = src.fetch()
    close = df["Close"].dropna()
    logret = src.log_returns_from_close(df)

    print(f"[DATA] {data_cfg.ticker} T={len(logret)} mean={float(logret.mean()):.9g} std={float(logret.std()):.9g} "
          f"kurt={float(kurtosis(logret, fisher=False, bias=False)):.3f}")

    # preprocess
    pre = LambertWPreprocessor(pre_cfg).fit(logret)
    r_train = pre.transform(logret)
    print("[PRE] state:", pre.summary())
    print("[PRE] r_train mean/std =", float(np.mean(r_train)), float(np.std(r_train)), "maxabs=", float(np.max(np.abs(r_train))))

    # dataset
    ds_builder = DatasetBuilder(ds_cfg)
    train_ds, X_windows, steps_per_epoch = ds_builder.build(r_train)
    print("[DS] windows", X_windows.shape, "mean=", float(X_windows.mean()), "std=", float(X_windows.std()))
    print("[DS] STEPS_PER_EPOCH=", steps_per_epoch)

    # targets
    real_windows = X_windows.astype(np.float32)
    N_TGT = min(int(run_cfg.n_target_windows), int(real_windows.shape[0]))
    idx = np.random.RandomState(ds_cfg.seed).choice(real_windows.shape[0], size=N_TGT, replace=False)
    real_sub = tf.constant(real_windows[idx], dtype=tf.float32)

    acf_abs_real_tgt = tf.reduce_mean(acf_tf(tf.abs(real_sub), run_cfg.vol_lags), axis=0)
    acf_sq_real_tgt = tf.reduce_mean(acf_tf(tf.square(real_sub), run_cfg.vol_lags), axis=0)
    lev_real_tgt = tf.reduce_mean(leverage_tf(real_sub, run_cfg.lev_lags), axis=0)

    print("[TGT] shapes:", acf_abs_real_tgt.shape, acf_sq_real_tgt.shape, lev_real_tgt.shape)

    evaluator = PaperEvaluator(real_series=logret, preproc=pre, train_cfg=train_cfg, dy_base_t=len(logret))
    print("[REAL STATS] mean=", evaluator.real_mean, "std=", evaluator.real_std, "q01,q05,q95,q99=", evaluator.real_qs)

    trainer = WGANGPTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        debug_cfg=dbg_cfg,
        window_len=ds_cfg.window_len,
        steps_per_epoch=steps_per_epoch,
        vol_lags=run_cfg.vol_lags,
        lev_lags=run_cfg.lev_lags,
        acf_abs_real_tgt=acf_abs_real_tgt,
        acf_sq_real_tgt=acf_sq_real_tgt,
        lev_real_tgt=lev_real_tgt,
    )

    run_out = trainer.fit(
        train_ds=train_ds,
        evaluator=evaluator,
        out_dir=out_dir_train,
        real_logret=logret,
        close_series=close,
        n_plot_runs=run_cfg.n_plot_runs,
        n_plot_paths=run_cfg.n_plot_paths,
        show_plots=run_cfg.show_plots,
        save_plots=run_cfg.save_plots,
        run_seed=run_cfg.seed,
    )

    print("\n=== RUN SUMMARY (WGAN-GP, no EMA, linear G output) ===")
    print("gen_type=", run_out["gen_type"])
    print(f"seed={run_out['seed']} best_score={run_out['best_score']:.6g} best_epoch={run_out['best_epoch']}")
    print(f"best_weights: {run_out['best_weights_path']}")


if __name__ == "__main__":
    main()

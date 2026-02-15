div.sourceCode {
    overflow: hidden;
}

# Wasserstein GANs for Realistic Financial Time-Series Generation

A modular Python package for generating realistic financial time series using Generative Adversarial Networks. The architecture is based on the **QuantGAN** framework by [Wiese et al. (2020)](#citation), implemented with **WGAN-GP** ([Gulrajani et al., 2017](#citation)) on top of **Wasserstein GAN** ([Arjovsky et al., 2017](#citation)).

## About

This package provides a production-ready implementation of temporal convolutional network (TCN) based generators for financial time series, with specialized preprocessing for heavy-tailed distributions and comprehensive evaluation metrics from the original paper.


### Key Features

- **Data Sources**:
  - **[DefeatBeta API](https://github.com/defeat-beta/defeatbeta-api)** (default, reliable)
  - **[yfinance](https://pypi.org/project/yfinance/)** (optional, less reliable)
  - Automatic CSV caching for fast repeated access

- **Two Generator Architectures**:
  - **Pure TCN**: Direct noise-to-returns mapping via temporal convolutions
  - **SVNN**: Stochastic Volatility Neural Network $r_t = \sigma_t \varepsilon_t + \mu_t$

- **Heavy-Tailed Distribution Handling**: Lambert-W transformation for fat-tailed financial returns

- **WGAN-GP Training**: Improved stability with gradient penalty

- **Comprehensive Metrics**: ACF, leverage correlation, and distribution distance (DY)

- **Modular Design**: Clean separation for easy extension and testing

## Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (optional, but recommended for training)

### Setup

```bash
# Clone the repository
git clone https://gitlab.ub.uni-giessen.de/J_Y5D5E8V/wasserstein-gans-for-realistic-financial-time-series-generation.git
cd quantgan

# Create and activate virtual environment (venv)
python3.10 -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# Update pip and install package
python -m pip install -U pip
pip install -e .

# Install Jupyter (for notebooks)
pip install -U ipykernel jupyterlab
python -m ipykernel install --user --name quantgan --display-name "quantgan (venv)"
```

### Dependencies

Core dependencies are automatically installed:
- TensorFlow >= 2.8.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Pandas >= 1.3.0
- defeatbeta-api 
- yfinance >= 0.2.0 

## Quick Start


### Train a Model

```python
from quantgan import ModelConfig, TrainConfig, DataConfig, PreprocessConfig, DatasetConfig
from quantgan.data import get_data_source, LambertWPreprocessor, DatasetBuilder
from quantgan.training import WGANGPTrainer
from quantgan.utils import set_all_seeds

# Configuration
set_all_seeds(0)
data_cfg = DataConfig(
    ticker="SPY",
    start="2009-05-01",
    end="2018-12-31",
    interval="1d",
)
model_cfg = ModelConfig(generator_type="pure_tcn")
train_cfg = TrainConfig(epochs=200)

# Load data - uses DefeatBeta API by default (open source, no rate limiting)
src = get_data_source(data_cfg)
df = src.fetch()
logret = src.log_returns_from_close(df)

# Preprocess with Lambert-W transformation
pre = LambertWPreprocessor(PreprocessConfig()).fit(logret)
r_train = pre.transform(logret)

# Create dataset
ds_builder = DatasetBuilder(DatasetConfig())
train_ds, _, steps_per_epoch = ds_builder.build(r_train)

# Train (see notebooks/01_training.ipynb for complete example)
trainer = WGANGPTrainer(...)
result = trainer.train(...)
```

**Note:** For a complete training example with all parameters, see `notebooks/01_training.ipynb`.

**Alternative data source (Yahoo Finance):**
```python
# To use Yahoo Finance instead (may have rate limiting/IP blocking issues)
data_cfg = DataConfig(ticker="SPY", source="yfinance", start="2009-05-01", end="2018-12-31")
src = get_data_source(data_cfg)
```

### Generate Synthetic Data

```python
from quantgan.utils import build_and_load_generator, generate_M_paths_raw

# Load trained model
netG = build_and_load_generator(
    model_cfg=model_cfg,
    window_len=127,
    weights_path="path/to/weights.h5",
    seed=0
)

# Generate 500 paths of length 4000
fake_paths = generate_M_paths_raw(
    netG=netG, preproc=pre, M=500, Ttilde=4000,
    window_len=127, z_dim=3, batch=50, seed=0
)
```

See `notebooks/` for detailed examples.

## Project Structure

```
quantgan/
├── quantgan/             # Main package
│   ├── config/           # Configuration classes
│   ├── data/             # Data loading & preprocessing
│   ├── models/           # Generator & Discriminator architectures
│   ├── training/         # WGAN-GP trainer
│   ├── evaluation/       # Metrics & visualization
│   └── utils/            # Helper functions
├── notebooks/            # Example notebooks
│   ├── 00_test_components.ipynb
│   ├── 01_training.ipynb
│   └── 02_evaluation.ipynb
├── data/                 # DefeatBeta/YFinance data cache
└── pyproject.toml
```

## Notebooks

The `notebooks/` directory contains step-by-step examples:

- **00_test_components.ipynb**: Test core components 
- **01_training.ipynb**: Complete training pipeline
- **02_evaluation.ipynb**: Model evaluation and visualization

## Data Sources

### DefeatBeta API (Default, Recommended)

The primary data source is the [DefeatBeta API](https://github.com/humandotlearning/defeatbeta-api), an open-source financial data provider that is used by default.

**Advantages:**
- Open source and transparent
- No rate limiting or IP blocking
- Reliable and consistent data quality
- Local CSV caching for fast repeated access

```python
from quantgan import DataConfig
from quantgan.data import get_data_source

# DefeatBeta is used automatically (default source)
data_cfg = DataConfig(
    ticker="SPY",
    start="2009-05-01",
    end="2018-12-31",
)
src = get_data_source(data_cfg)
df = src.fetch()  # Downloads once, then uses cached CSV
logret = src.log_returns_from_close(df)
```

### Yahoo Finance (Alternative)

Yahoo Finance is available as an alternative, but has known limitations (e. g. frequent rate limit and IP blocking after too many requests):

To use Yahoo Finance, set `source="yfinance"` in your config:

```python
from quantgan import DataConfig
from quantgan.data import get_data_source

# Explicitly use Yahoo Finance
data_cfg = DataConfig(
    ticker="SPY",
    start="2009-05-01",
    end="2018-12-31",
    source="yfinance",  # Override default
)
src = get_data_source(data_cfg)
df = src.fetch()
logret = src.log_returns_from_close(df)
```

**Note:** The `get_data_source()` factory function automatically selects the appropriate data source based on your config, making it easy to switch between sources without changing your code.

## Generator Types

### Pure TCN
Standard temporal convolutional network that directly maps latent noise to returns.

```python
ModelConfig(generator_type="pure_tcn")
```

### SVNN (Stochastic Volatility Neural Network)
Models returns as r_t = σ_t × ε_t + μ_t, where σ_t (volatility) and μ_t (drift) are predicted by a TCN based on past latent variables.

```python
ModelConfig(generator_type="svnn")
```

## Citation

This implementation is based on the following papers:

### QuantGAN Framework
```bibtex
@article{wiese2020quant,
  title={Quant GANs: Deep Generation of Financial Time Series},
  author={Wiese, Magnus and Knobloch, Robert and Korn, Ralf and Kretschmer, Peter},
  journal={Quantitative Finance},
  volume={20},
  number={9},
  pages={1419--1440},
  year={2020},
  publisher={Taylor \& Francis}
}
```

### WGAN-GP 
```bibtex
@inproceedings{gulrajani2017improved,
  title={Improved Training of Wasserstein GANs},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5767--5777},
  year={2017}
}
```

### Wasserstein GAN
```bibtex
@inproceedings{arjovsky2017wasserstein,
  title={Wasserstein GAN},
  author={Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  booktitle={International Conference on Machine Learning},
  pages={214--223},
  year={2017},
  organization={PMLR}
}
```

If you use this code in your research, please cite the relevant papers.


## License

MIT License - see LICENSE file for details.



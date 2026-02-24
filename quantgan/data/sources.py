"""Data sources for financial time series."""

import os
from pathlib import Path
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseDataSource(ABC):
    """Base class for data sources with CSV caching."""

    source_name = "base"  # Override in subclasses

    def __init__(self, cfg, data_dir="data"):
        """Initialize data source.
        
        Args:
            cfg: DataConfig instance
            data_dir: Directory to store/load CSV files (default: "data")
        """
        self.cfg = cfg
        self.data_dir = Path(data_dir)
    
    def _get_csv_path(self):
        """Generate CSV filename based on config.
        
        Returns:
            Path object for the CSV file
        """
        filename = f"{self.source_name}_{self.cfg.ticker}_{self.cfg.start}_{self.cfg.end}_{self.cfg.interval}.csv"
        return self.data_dir / filename
    
    def _load_from_csv(self, csv_path):
        """Load data from CSV cache.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            Exception: If loading fails
        """
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"[DATA] Loaded from cache: {csv_path}")
        return df
    
    def _save_to_csv(self, df, csv_path):
        """Save DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            csv_path: Path object where to save
        """
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path)
            print(f"[DATA] Saved to cache: {csv_path}")
        except Exception as e:
            print(f"[DATA] Warning: Failed to save CSV ({e})")
    
    @abstractmethod
    def _download_data(self):
        """Download data from the specific source.
        
        Must be implemented by subclasses.
        
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    def fetch(self, force_download=False):
        """Fetch data from CSV cache or download from source.
        
        Args:
            force_download: If True, ignore cache and download from source
            
        Returns:
            DataFrame with OHLCV data
        """
        csv_path = self._get_csv_path()
        
        # Try to load from CSV first (unless forced to download)
        if not force_download and csv_path.exists():
            try:
                return self._load_from_csv(csv_path)
            except Exception as e:
                print(f"[DATA] Warning: Failed to load CSV ({e}), downloading...")
        
        # Download from source
        df = self._download_data()
        
        # Save to CSV for future use
        self._save_to_csv(df, csv_path)
        
        return df
    
    @staticmethod
    def log_returns_from_close(df):
        """Compute log returns from close prices.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Log returns as numpy array
        """
        close = df["Close"].dropna()
        logret = np.log(close / close.shift(1)).dropna().values.astype(np.float64)
        if len(logret) <= 0:
            raise ValueError("[DATA] logret is empty.")
        return logret


class DefeatBetaSource(BaseDataSource):
    """DefeatBeta API data source with CSV caching."""
    
    source_name = "defeatbeta"
    
    def _download_data(self):
        """Download data from DefeatBeta API.
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"[DATA] Downloading {self.cfg.ticker} from DefeatBeta API...")
        
        try:
            from defeatbeta_api.data.ticker import Ticker
            from defeatbeta_api.client.duckdb_conf import Configuration
        except ImportError:
            raise ImportError(
                "defeatbeta-api not installed. Run: pip install defeatbeta-api"
            )

        db_cfg = Configuration()
        db_cfg.cache_httpfs_type = "noop"
        ticker = Ticker(self.cfg.ticker, config=db_cfg)

        df_raw = ticker.price()
        
        if df_raw is None or len(df_raw) == 0:
            raise ValueError(
                "[DefeatBeta] Download returned empty dataframe. "
                "Check network / ticker."
            )
        
        # Convert to yfinance-compatible format
        df = df_raw.copy()
        
        # Parse report_date: force UTC, then remove timezone
        df["report_date"] = pd.to_datetime(df["report_date"], utc=True, errors="coerce")
        
        # Remove rows with invalid dates
        if df["report_date"].isna().any():
            n_bad = df["report_date"].isna().sum()
            print(f"[DATA] Warning: {n_bad} rows with invalid dates removed")
            df = df.loc[~df["report_date"].isna()].copy()
        
        df["report_date"] = df["report_date"].dt.tz_convert(None)
        df = df.set_index("report_date")
        
        # Rename columns to match yfinance
        df = df.rename(columns={
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume"
        })
        
        # Sort by date
        df = df.sort_index()
        
        # Filter to desired time range
        df = df.loc[self.cfg.start:self.cfg.end]
        
        if len(df) == 0:
            raise ValueError(
                f"[DefeatBeta] No data in range {self.cfg.start} to {self.cfg.end}"
            )
        
        if "Close" not in df.columns:
            raise ValueError(f"[DefeatBeta] 'Close' not in columns: {list(df.columns)}")
        
        return df


class YFinanceSource(BaseDataSource):
    """Yahoo Finance data source with CSV caching."""
    
    source_name = "yfinance"
    
    def _adjust_end_date(self, end_date_str):
        """Add one day to end_date for yfinance (it excludes the last day).
        
        Args:
            end_date_str: Date string in format "YYYY-MM-DD"
            
        Returns:
            Adjusted date string
        """
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        end_date_adjusted = end_date + timedelta(days=1)
        return end_date_adjusted.strftime("%Y-%m-%d")
    
    def _download_data(self):
        """Download data from Yahoo Finance.
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. Run: pip install yfinance"
            )
        
        # Download from yfinance with adjusted end_date
        end_date_adjusted = self._adjust_end_date(self.cfg.end)
        print(f"[DATA] Downloading {self.cfg.ticker} from Yahoo Finance...")
        print(f"[DATA] Note: Requesting until {end_date_adjusted} to include {self.cfg.end}")
        
        tk = yf.Ticker(self.cfg.ticker)
        df = tk.history(
            start=self.cfg.start,
            end=end_date_adjusted,
            interval=self.cfg.interval,
            auto_adjust=False,
            actions=True,
        )
        
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df = df.sort_index()

        if df is None or len(df) == 0:
            raise ValueError(
                "[YF] Download returned empty dataframe. "
                "Check network / rate limit / ticker."
            )
        if "Close" not in df.columns:
            raise ValueError(f"[YF] 'Close' not in columns: {list(df.columns)}")
        
        return df


def get_data_source(cfg, data_dir="data"):
    """Factory function to get the appropriate data source.
    
    Args:
        cfg: DataConfig instance
        data_dir: Directory to store/load CSV files
        
    Returns:
        Data source instance (DefeatBetaSource or YFinanceSource)
    """
    if cfg.source == "yfinance":
        return YFinanceSource(cfg, data_dir)
    elif cfg.source == "defeatbeta":
        return DefeatBetaSource(cfg, data_dir)
    else:
        raise ValueError(
            f"Unknown source: {cfg.source}. Must be 'defeatbeta' or 'yfinance'"
        )

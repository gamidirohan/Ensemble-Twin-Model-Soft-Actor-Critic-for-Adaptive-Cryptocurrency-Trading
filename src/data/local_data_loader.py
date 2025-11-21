"""
Local Data Loader for Parquet Files
Loads cryptocurrency data from local Binance parquet files instead of using API
"""
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility
from pathlib import Path

class LocalDataLoader:
    def __init__(self, data_dir=r"C:\Projects\Binance Dataset"):
        """
        Initialize local data loader
        
        Args:
            data_dir: Path to directory containing .parquet files
        """
        self.data_dir = Path(data_dir)
        
    def load_parquet(self, filename):
        """
        Load a single parquet file
        
        Args:
            filename: Name of parquet file (e.g., '1INCH-BTC.parquet')
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle Binance parquet format with 'open_time' column
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
        else:
            # Create a synthetic timestamp if none exists
            print("Warning: No timestamp found, creating synthetic timestamps")
            df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='1h')
            
        print(f"Loaded {len(df)} rows from {filename}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def add_indicators(self, df):
        """
        Add technical indicators to the dataframe
        Same as BinanceDataFetcher.add_indicators()
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # RSI
        df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pband'] = bb.bollinger_pband()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Drop NaN rows
        df.dropna(inplace=True)
        
        print(f"Added technical indicators. Remaining rows after dropna: {len(df)}")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for RL training
        Same as BinanceDataFetcher.prepare_features()
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            numpy array of normalized features
        """
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi', 'macd', 'macd_signal', 'macd_diff',
                       'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pband']
        
        features = df[feature_cols].copy()
        
        # Normalize price-based features
        if len(features) > 0 and 'close' in features.columns:
            first_close = features['close'].iloc[0] if len(features['close']) > 0 else 1.0
            for col in ['open', 'high', 'low', 'close']:
                if col in features.columns:
                    features[col] = features[col] / first_close
        
        # Normalize volume
        if 'volume' in features.columns:
            vol_mean = features['volume'].mean()
            if vol_mean > 0:
                features['volume'] = features['volume'] / vol_mean
        
        # Convert to numpy array
        feature_array = features.values.astype(np.float32)
        
        print(f"Prepared features shape: {feature_array.shape}")
        
        return feature_array
    
    def load_and_prepare(self, filename, limit=None):
        """
        Load parquet file and prepare all features in one go
        
        Args:
            filename: Name of parquet file
            limit: Optional limit on number of rows (most recent)
            
        Returns:
            tuple: (df_with_indicators, features_array)
        """
        df = self.load_parquet(filename)
        
        # Limit to most recent data if specified
        if limit and len(df) > limit:
            df = df.tail(limit).copy()
            print(f"Limited to most recent {limit} rows")
        
        df = self.add_indicators(df)
        features = self.prepare_features(df)
        
        return df, features


if __name__ == "__main__":
    # Test the loader
    loader = LocalDataLoader()
    
    print("\n" + "="*60)
    print("Testing Local Data Loader with 1INCH-BTC.parquet")
    print("="*60 + "\n")
    
    try:
        df, features = loader.load_and_prepare('1INCH-BTC.parquet', limit=2000)
        
        print("\n" + "="*60)
        print("Data loaded successfully!")
        print("="*60)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Features shape: {features.shape}")
        print(f"\nFirst few rows of data:")
        print(df.head())
        print(f"\nData statistics:")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Data Acquisition and Preprocessing Module for Dynamic Data Labeling
Implements fractional differentiation, feature engineering, and robust data processing
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta
import pandas_ta as pta

from config import DataConfig


class FractionalDifferentiation:
    """
    Implements Fractional Differentiation to achieve stationarity while preserving memory
    Based on the methodology from Advances in Financial Machine Learning
    """
    
    @staticmethod
    def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
        """
        Compute weights for fractional differentiation with fixed window
        
        Args:
            d: Fractional differentiation order
            thres: Threshold for weight truncation
            
        Returns:
            weights: Array of weights for fractional differentiation
        """
        w = [1.0]
        k = 1
        
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
            
        return np.array(w[::-1])
    
    @staticmethod
    def fracDiff_FFD(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
        """
        Apply fractional differentiation with fixed window
        
        Args:
            series: Time series to differentiate
            d: Fractional differentiation order
            thres: Threshold for weight truncation
            
        Returns:
            Fractionally differentiated series
        """
        weights = FractionalDifferentiation.get_weights_ffd(d, thres)
        width = len(weights) - 1
        
        if width == 0:
            return series.copy()
        
        # Apply convolution
        df = {}
        for name in series.index[width:]:
            loc0 = series.index.get_loc(name)
            if not np.isfinite(series.iloc[loc0]):
                continue
            df[name] = np.dot(weights.T, series.iloc[loc0-width:loc0+1].values)
        
        df = pd.Series(df, index=series.index[width:])
        return df
    
    @staticmethod
    def find_min_ffd_order(series: pd.Series, max_d: float = 1.0, 
                          step: float = 0.01, pvalue_thresh: float = 0.05) -> float:
        """
        Find minimum fractional differentiation order for stationarity
        
        Args:
            series: Time series to test
            max_d: Maximum d value to test
            step: Step size for d
            pvalue_thresh: p-value threshold for ADF test
            
        Returns:
            Minimum d value that achieves stationarity
        """
        d_values = np.arange(0, max_d + step, step)
        
        for d in d_values:
            if d == 0:
                diff_series = series
            else:
                diff_series = FractionalDifferentiation.fracDiff_FFD(series, d)
            
            if len(diff_series.dropna()) < 10:
                continue
                
            adf_result = adfuller(diff_series.dropna(), autolag='AIC')
            if adf_result[1] < pvalue_thresh:
                return d
        
        return max_d


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculation
    """
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive set of technical indicators
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with all technical indicators
        """
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Trend Indicators
        data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
        data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
        data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
        
        # MACD
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
        data['MACD_hist'] = ta.trend.macd(data['Close'])
        
        # ADX
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        data['ADX_pos'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'])
        data['ADX_neg'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'])
        
        # Bollinger Bands
        data['BB_high'] = ta.volatility.bollinger_hband(data['Close'])
        data['BB_low'] = ta.volatility.bollinger_lband(data['Close'])
        data['BB_mid'] = ta.volatility.bollinger_mavg(data['Close'])
        data['BB_width'] = ta.volatility.bollinger_wband(data['Close'])
        data['BB_pband'] = ta.volatility.bollinger_pband(data['Close'])
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['Stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
        data['Stoch_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
        data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
        data['ROC'] = ta.momentum.roc(data['Close'])
        data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
        
        # Volume Indicators
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['VWAP'] = ta.volume.volume_weighted_average_price(
            data['High'], data['Low'], data['Close'], data['Volume']
        )
        data['Volume_SMA'] = ta.volume.sma_ease_of_movement(data['High'], data['Low'], data['Volume'])
        
        # Volatility Indicators
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        data['Keltner_high'] = ta.volatility.keltner_channel_hband(data['High'], data['Low'], data['Close'])
        data['Keltner_low'] = ta.volatility.keltner_channel_lband(data['High'], data['Low'], data['Close'])
        
        # Price-based features
        data['HL_ratio'] = (data['High'] - data['Low']) / data['Close']
        data['OC_ratio'] = (data['Open'] - data['Close']) / data['Close']
        data['Price_change'] = data['Close'].pct_change()
        data['Volume_change'] = data['Volume'].pct_change()
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
            data[f'Return_lag_{lag}'] = data['Price_change'].shift(lag)
        
        return data


class DataProcessor:
    """
    Main data processing class for dynamic labeling system
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.ffd = FractionalDifferentiation()
        self.tech_indicators = TechnicalIndicators()
        self.scalers = {}
        
    def download_data(self, symbols: List[str], 
                     start_date: str = None, 
                     end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Download financial data using yfinance with improved error handling
        
        Args:
            symbols: List of symbols to download
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        import time
        
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        
        data = {}
        
        # Try downloading all symbols at once first (more efficient)
        try:
            print(f"Downloading data for {len(symbols)} symbols: {symbols}")
            tickers = yf.Tickers(' '.join(symbols))
            hist_data = tickers.history(
                start=start_date, 
                end=end_date, 
                interval=self.config.interval,
                group_by='ticker'
            )
            
            for symbol in symbols:
                try:
                    if symbol in hist_data.columns.levels[0]:
                        df = hist_data[symbol].dropna()
                        if len(df) > 20:  # Minimum data requirement
                            # Clean column names
                            df.columns = [col.replace(' ', '_') for col in df.columns]
                            data[symbol] = df
                            print(f"✅ Downloaded {len(df)} records for {symbol}")
                        else:
                            print(f"⚠️ Insufficient data for {symbol} ({len(df)} records)")
                    else:
                        print(f"❌ No data found for {symbol}")
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {str(e)}")
                    
        except Exception as e:
            print(f"Bulk download failed: {str(e)}")
            print("Trying individual downloads with delays...")
            
            # Fallback: download individually with delays
            for i, symbol in enumerate(symbols):
                try:
                    if i > 0:  # Add delay between requests
                        time.sleep(1)
                    
                    print(f"Downloading {symbol}... ({i+1}/{len(symbols)})")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date, 
                        end=end_date, 
                        interval=self.config.interval
                    )
                    
                    if len(df) > 20:  # Minimum data requirement
                        # Clean column names
                        df.columns = [col.replace(' ', '_') for col in df.columns]
                        df = df.dropna()
                        data[symbol] = df
                        print(f"✅ Downloaded {len(df)} records for {symbol}")
                    else:
                        print(f"⚠️ Insufficient data for {symbol}")
                        
                except Exception as e:
                    print(f"❌ Error downloading {symbol}: {str(e)}")
                    continue
                    
        print(f"\n📊 Successfully downloaded data for {len(data)} out of {len(symbols)} symbols")
        return data
    
    def add_external_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add external market features like VIX
        
        Args:
            df: Main dataframe
            symbol: Symbol being processed
            
        Returns:
            DataFrame with external features
        """
        data = df.copy()
        
        try:
            # Download VIX data
            vix = yf.download("^VIX", 
                            start=df.index[0], 
                            end=df.index[-1], 
                            interval=self.config.interval)
            
            if not vix.empty:
                vix_close = vix['Close'].reindex(df.index, method='ffill')
                data['VIX'] = vix_close
                data['VIX_change'] = vix_close.pct_change()
        
        except Exception as e:
            print(f"Could not add VIX data: {str(e)}")
            
        # Add market day features
        data['DayOfWeek'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
        
        # Add time-based features
        data['Days_since_start'] = (data.index - data.index[0]).days
        
        return data
    
    def apply_fractional_differentiation(self, df: pd.DataFrame, 
                                       price_columns: List[str] = None) -> pd.DataFrame:
        """
        Apply fractional differentiation to price series
        
        Args:
            df: Input dataframe
            price_columns: Columns to apply FFD to
            
        Returns:
            DataFrame with fractionally differentiated features
        """
        if price_columns is None:
            price_columns = ['Open', 'High', 'Low', 'Close']
        
        data = df.copy()
        
        for col in price_columns:
            if col in data.columns:
                # Find optimal d value
                d_opt = self.ffd.find_min_ffd_order(
                    data[col], 
                    pvalue_thresh=self.config.adf_pvalue_threshold
                )
                
                # Apply fractional differentiation
                if d_opt > 0:
                    ffd_series = self.ffd.fracDiff_FFD(data[col], d_opt, self.config.ffd_threshold)
                    data[f'{col}_FFD'] = ffd_series
                    print(f"Applied FFD with d={d_opt:.3f} to {col}")
                else:
                    data[f'{col}_FFD'] = data[col]
                    print(f"No FFD needed for {col}")
        
        return data
    
    def rolling_standardization(self, df: pd.DataFrame, 
                              feature_columns: List[str]) -> pd.DataFrame:
        """
        Apply rolling standardization to avoid look-ahead bias
        
        Args:
            df: Input dataframe
            feature_columns: Columns to standardize
            
        Returns:
            Standardized dataframe
        """
        data = df.copy()
        window = self.config.standardization_window
        
        for col in feature_columns:
            if col in data.columns:
                # Calculate rolling mean and std
                rolling_mean = data[col].rolling(window=window, min_periods=10).mean()
                rolling_std = data[col].rolling(window=window, min_periods=10).std()
                
                # Standardize
                data[f'{col}_std'] = (data[col] - rolling_mean) / rolling_std
        
        return data
    
    def create_sequences(self, df: pd.DataFrame, 
                        feature_columns: List[str],
                        sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            df: Input dataframe
            feature_columns: Features to include
            sequence_length: Length of sequences
            
        Returns:
            X: Feature sequences
            indices: Corresponding indices
        """
        sequence_length = sequence_length or self.config.lookback_window
        
        # Select and clean features
        features = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Create sequences
        X = []
        indices = []
        
        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            indices.append(features.index[i])
        
        return np.array(X), np.array(indices)
    
    def process_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Complete processing pipeline for a single symbol
        
        Args:
            symbol: Symbol being processed
            df: Raw OHLCV data
            
        Returns:
            Processed data dictionary
        """
        print(f"\nProcessing {symbol}...")
        
        # Step 1: Add technical indicators
        df_with_indicators = self.tech_indicators.calculate_all_indicators(df)
        print(f"Added technical indicators, shape: {df_with_indicators.shape}")
        
        # Step 2: Add external features
        df_with_external = self.add_external_features(df_with_indicators, symbol)
        print(f"Added external features, shape: {df_with_external.shape}")
        
        # Step 3: Apply fractional differentiation
        df_with_ffd = self.apply_fractional_differentiation(df_with_external)
        print(f"Applied FFD, shape: {df_with_ffd.shape}")
        
        # Step 4: Get feature columns (exclude OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
        feature_columns = [col for col in df_with_ffd.columns if col not in exclude_cols]
        
        # Step 5: Apply rolling standardization
        df_standardized = self.rolling_standardization(df_with_ffd, feature_columns)
        print(f"Applied standardization, shape: {df_standardized.shape}")
        
        # Step 6: Select final features (standardized versions)
        final_features = [col for col in df_standardized.columns if col.endswith('_std')]
        
        # If no standardized features, use original features
        if not final_features:
            final_features = feature_columns
        
        # Step 7: Create sequences
        X, indices = self.create_sequences(df_standardized, final_features)
        
        print(f"Created sequences: {X.shape}")
        
        # Prepare output
        result = {
            'features': X,
            'indices': indices,
            'raw_data': df,
            'processed_data': df_standardized,
            'feature_names': final_features,
            'ohlcv': df_standardized[['Open', 'High', 'Low', 'Close', 'Volume']].reindex(indices)
        }
        
        return result
    
    def process_all_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all configured symbols
        
        Returns:
            Dictionary of processed data for all symbols
        """
        all_symbols = self.config.stock_symbols + self.config.crypto_symbols
        
        # Download data
        print("Downloading data...")
        raw_data = self.download_data(all_symbols, self.config.start_date, self.config.end_date)
        
        # Process each symbol
        processed_data = {}
        for symbol, df in raw_data.items():
            try:
                processed_data[symbol] = self.process_symbol(symbol, df)
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        print(f"\nSuccessfully processed {len(processed_data)} symbols")
        return processed_data


if __name__ == "__main__":
    from config import load_config
    
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = DataProcessor(config.data)
    
    # Process all data
    processed_data = processor.process_all_data()
    
    # Save processed data
    import pickle
    import os
    
    os.makedirs("data", exist_ok=True)
    with open("data/processed_data.pkl", "wb") as f:
        pickle.dump(processed_data, f)
    
    print("Data processing complete!")
    print(f"Processed data saved to data/processed_data.pkl")
    
    # Print summary
    for symbol, data in processed_data.items():
        print(f"{symbol}: {data['features'].shape} sequences, {len(data['feature_names'])} features")

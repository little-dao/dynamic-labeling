"""
Configuration file for Dynamic Data Labeling System
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class DataConfig:
    """Configuration for data acquisition and preprocessing"""
    # Stock symbols to analyze
    stock_symbols: List[str] = None
    crypto_symbols: List[str] = None
    
    # Data source parameters
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    interval: str = "1d"  # 1d, 1h, 5m, etc.
    
    # Feature engineering
    technical_indicators: List[str] = None
    lookback_window: int = 60
    
    # Fractional differentiation
    ffd_threshold: float = 0.01  # Threshold for FFD
    adf_pvalue_threshold: float = 0.05  # p-value threshold for ADF test
    
    # Standardization
    standardization_window: int = 252  # Rolling window for standardization
    
    def __post_init__(self):
        if self.stock_symbols is None:
            self.stock_symbols = ["AAPL", "MSFT", "GOOGL"]  # Reduced to avoid rate limits
        
        if self.crypto_symbols is None:
            self.crypto_symbols = ["BTC-USD"]  # Reduced to avoid rate limits
        
        if self.technical_indicators is None:
            self.technical_indicators = [
                "RSI", "MACD", "ADX", "BB", "SMA", "EMA", 
                "STOCH", "CCI", "Williams", "ROC", "OBV", "VWAP"
            ]


@dataclass
class ModelConfig:
    """Configuration for LSTM model architecture"""
    # Model architecture
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    
    # CNN components
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # Attention mechanism
    attention_dim: int = 64
    use_attention: bool = True
    
    # Output layer
    output_dim: int = 3  # PT, SL, Time Horizon
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 15
    
    # Loss function weights
    pt_weight: float = 1.0
    sl_weight: float = 1.0
    th_weight: float = 1.0
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 128]
        
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5, 7]


@dataclass
class TradingConfig:
    """Configuration for trading simulation and labeling"""
    # Profit-taking and stop-loss ranges
    pt_min: float = 0.01  # 1%
    pt_max: float = 0.15  # 15%
    sl_min: float = 0.005  # 0.5%
    sl_max: float = 0.10   # 10%
    
    # Time horizon ranges (in periods)
    th_min: int = 1
    th_max: int = 30
    
    # Trading parameters
    transaction_cost: float = 0.001  # 0.1%
    initial_capital: float = 100000.0
    
    # Meta-labeling parameters
    confidence_threshold: float = 0.6
    rebalance_frequency: int = 5  # days


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and robustness testing"""
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Performance metrics
    risk_free_rate: float = 0.02  # 2% annual
    
    # Robustness testing
    noise_levels: List[float] = None
    monte_carlo_runs: int = 1000
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    trading: TradingConfig = None
    evaluation: EvaluationConfig = None
    
    # General settings
    random_seed: int = 42
    device: str = "cuda"  # cuda, cpu, mps
    num_workers: int = 4
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)


def load_config() -> Config:
    """Load default configuration"""
    return Config()

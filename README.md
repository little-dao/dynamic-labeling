# Dynamic Data Labeling for Stock Prediction

A sophisticated deep learning framework that uses LSTM networks to dynamically generate profit-taking thresholds, stop-loss thresholds, and optimal time horizons for stock trades. This system moves beyond traditional static labeling methods by leveraging the power of neural networks to predict crucial trading parameters in real-time.

## 🚀 Key Features

- **Dynamic Parameter Prediction**: LSTM-based model predicts profit-taking (PT), stop-loss (SL), and time horizon (TH) for each trade
- **Fractional Differentiation**: Advanced preprocessing to achieve stationarity while preserving memory
- **Meta-Labeling System**: Adaptive feedback loop that learns from trade outcomes
- **Comprehensive Evaluation**: Financial performance metrics and robustness testing
- **Attention Mechanisms**: Enhanced LSTM with multi-head attention for better temporal modeling
- **Real-time Adaptation**: Continuous learning from market feedback

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dynamic-labeling.git
cd dynamic-labeling
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data models results logs
```

## 🚀 Quick Start

### Basic Usage

Run the complete pipeline with default settings:

```bash
python main_pipeline.py
```

### Step-by-Step Execution

Run specific stages of the pipeline:

```bash
# Data processing only
python main_pipeline.py --stages data

# Data processing and label generation
python main_pipeline.py --stages data labels

# Full pipeline with state saving
python main_pipeline.py --save-state pipeline_state.pkl
```

### Individual Components

Test individual components:

```bash
# Data processing
python data_processor.py

# Model testing
python dynamic_lstm_model.py

# Label generation
python labeling_system.py

# Evaluation metrics
python evaluation_metrics.py
```

## 🏗 Architecture Overview

The system consists of five main components:

### 1. Data Processing Module (`data_processor.py`)
- Downloads financial data using yfinance
- Implements fractional differentiation for stationarity
- Generates comprehensive technical indicators
- Applies rolling standardization to avoid look-ahead bias

### 2. Dynamic LSTM Model (`dynamic_lstm_model.py`)
- LSTM-CNN encoder-decoder architecture
- Multi-head attention mechanism
- Predicts three continuous values: PT%, SL%, and time horizon
- Custom loss function with consistency constraints

### 3. Labeling System (`labeling_system.py`)
- Trade simulation based on predicted parameters
- Meta-labeling with adaptive feedback
- Multiple label types: binary, continuous, multi-class
- Performance-based label adaptation

### 4. Evaluation Module (`evaluation_metrics.py`)
- Comprehensive financial performance metrics
- Robustness testing with noise simulation
- Cross-validation for temporal data
- Interactive visualizations

### 5. Main Pipeline (`main_pipeline.py`)
- Orchestrates the complete workflow
- Handles data flow between components
- Provides state management and logging
- Command-line interface

## ⚙️ Configuration

The system uses a hierarchical configuration system in `config.py`:

### Data Configuration
```python
@dataclass
class DataConfig:
    stock_symbols: List[str] = ["AAPL", "AMZN", "KO", "SBUX", "TSLA"]
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    lookback_window: int = 60
    ffd_threshold: float = 0.01
```

### Model Configuration
```python
@dataclass
class ModelConfig:
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    learning_rate: float = 0.001
    batch_size: int = 64
    use_attention: bool = True
```

### Trading Configuration
```python
@dataclass
class TradingConfig:
    pt_min: float = 0.01  # 1%
    pt_max: float = 0.15  # 15%
    sl_min: float = 0.005  # 0.5%
    sl_max: float = 0.10   # 10%
```

## 📊 Usage Examples

### Example 1: Basic Training

```python
from config import load_config
from main_pipeline import DynamicLabelingPipeline

# Load configuration
config = load_config()

# Initialize pipeline
pipeline = DynamicLabelingPipeline(config)

# Run complete pipeline
results = pipeline.run_complete_pipeline()

print(f"Final Sharpe Ratio: {results['evaluate']['performance_metrics']['sharpe_ratio']:.4f}")
```

### Example 2: Custom Configuration

```python
from config import Config, DataConfig, ModelConfig

# Custom configuration
data_config = DataConfig(
    stock_symbols=["GOOGL", "MSFT", "NVDA"],
    start_date="2021-01-01",
    lookback_window=30
)

model_config = ModelConfig(
    lstm_hidden_size=256,
    learning_rate=0.0005,
    use_attention=True
)

config = Config(data=data_config, model=model_config)

# Run with custom config
pipeline = DynamicLabelingPipeline(config)
results = pipeline.run_complete_pipeline()
```

### Example 3: Robustness Analysis

```python
from evaluation_metrics import RobustnessAnalyzer
from config import load_config

config = load_config()
analyzer = RobustnessAnalyzer(config.evaluation)

# Evaluate robustness
robustness_results = analyzer.evaluate_labeling_robustness(trades, labels)
print(f"Robustness Score: {robustness_results['robustness_score']:.4f}")
```

## 🧠 Model Architecture

### LSTM-CNN Encoder-Decoder

The core model combines several advanced architectures:

```
Input Features (seq_len, features)
    ↓
Positional Encoding
    ↓
Bidirectional LSTM Layers
    ↓
Multi-Head Attention
    ↓
CNN Feature Extraction
    ↓
Feature Fusion
    ↓
Three Output Heads (PT, SL, TH)
```

### Key Components

1. **Positional Encoding**: Helps the model understand temporal relationships
2. **Bidirectional LSTM**: Captures both forward and backward temporal dependencies
3. **Attention Mechanism**: Focuses on important time steps and features
4. **CNN Component**: Extracts spatial patterns from feature sequences
5. **Multi-Task Learning**: Jointly optimizes three related prediction tasks

### Loss Function

Custom loss function combining:
- Mean Squared Error for each prediction
- Consistency constraint (PT > SL)
- Weighted combination based on financial importance

## 📈 Evaluation Metrics

### Financial Performance Metrics

- **Return Metrics**: Total return, annualized return, cumulative return
- **Risk Metrics**: Volatility, maximum drawdown, Value at Risk
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade-Based**: Win rate, profit factor, hit ratio

### Robustness Testing

- **Noise Simulation**: Tests label quality under different accuracy levels
- **Monte Carlo Analysis**: Statistical significance testing
- **Cross-Validation**: Time series aware validation
- **Comparative Analysis**: Performance vs. traditional methods

### Visualization

Interactive HTML reports including:
- Performance comparison charts
- Robustness analysis plots
- Trade distribution analysis
- Cumulative return curves

## 🔄 Adaptive Learning

The system implements several adaptive mechanisms:

### Meta-Labeling
- Generates labels based on actual trade outcomes
- Adapts to changing market conditions
- Filters false positive signals

### Feedback Loop
- Continuous model improvement
- Performance-based retraining triggers
- Dynamic parameter adjustment

### Robustness Optimization
- Monitors label quality degradation
- Automatic model rebalancing
- Noise-resistant training

## 📁 Project Structure

```
dynamic-labeling/
├── config.py                 # Configuration management
├── data_processor.py         # Data acquisition and preprocessing
├── dynamic_lstm_model.py     # LSTM model architecture
├── labeling_system.py        # Meta-labeling and feedback
├── evaluation_metrics.py     # Performance evaluation
├── main_pipeline.py          # Main execution pipeline
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── data/                     # Data storage
├── models/                   # Trained models
├── results/                  # Output visualizations
└── logs/                     # Execution logs
```

## 🧪 Testing

Run the test suite:

```bash
# Test individual components
python -m pytest tests/

# Test with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on concepts from "Advances in Financial Machine Learning" by Marcos López de Prado
- Inspired by recent advances in attention mechanisms and meta-learning
- Built using PyTorch, pandas, and the broader Python scientific computing ecosystem

## 📚 Further Reading

- [Fractional Differentiation in Financial Time Series](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Meta-Labeling in Financial Machine Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816)
- [Attention Mechanisms in Deep Learning](https://arxiv.org/abs/1706.03762)

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review the example notebooks in `/examples`

---

**Note**: This system is for research and educational purposes. Always perform thorough backtesting and risk assessment before using in live trading environments.
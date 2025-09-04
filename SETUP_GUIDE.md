# Setup Guide - Dynamic Data Labeling for Stock Prediction

## Quick Start Instructions

### 1. Environment Setup

```bash
# Clone or download the project
cd dynamic-labeling

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues with torch-audio, try:
pip install torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if you have CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation

```bash
# Test the installation with a simple check
python -c "import torch; import pandas; import yfinance; print('All packages installed successfully!')"
```

### 4. Running the System

#### Option A: Jupyter Notebook (Recommended for Research)
```bash
# Start Jupyter
jupyter notebook

# Open Dynamic_Data_Labeling_Research.ipynb
# Run cells sequentially to see the complete implementation
```

#### Option B: Command Line Execution
```bash
# Run complete pipeline
python main_pipeline.py

# Run specific stages
python main_pipeline.py --stages data labels model train evaluate

# Run individual components for testing
python data_processor.py
python dynamic_lstm_model.py
python labeling_system.py
python evaluation_metrics.py
```

#### Option C: Example Scripts
```bash
# Run example usage scenarios
python example_usage.py
```

### 5. Project Structure After Setup

```
dynamic-labeling/
├── architecture_diagram.md      # System architecture documentation
├── config.py                   # Configuration management
├── data_processor.py           # Data processing and FFD implementation
├── dynamic_lstm_model.py       # LSTM model architecture
├── evaluation_metrics.py       # Performance evaluation framework
├── example_usage.py            # Example usage scenarios
├── labeling_system.py          # Meta-labeling system
├── main_pipeline.py            # Main execution pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── SETUP_GUIDE.md              # This file
├── Dynamic_Data_Labeling_Research.ipynb  # Complete research notebook
├── data/                       # Generated data files
├── models/                     # Saved model checkpoints
├── results/                    # Output visualizations
└── logs/                       # Execution logs
```

## Common Issues and Solutions

### Issue 1: Package Installation Errors
```bash
# If pip install fails, try upgrading pip first:
python -m pip install --upgrade pip

# For Apple Silicon Macs, you might need:
pip install --upgrade pip setuptools wheel
```

### Issue 2: CUDA/GPU Issues
```bash
# Check if CUDA is available:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If you want CPU-only (safer for beginners):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Financial Data Download Issues
- The system uses `yfinance` which may occasionally have rate limits
- If data download fails, try again after a few minutes
- For production use, consider using professional data providers

### Issue 4: Memory Issues
- The default configuration is designed for systems with 8GB+ RAM
- For lower memory systems, reduce batch size in `config.py`:
  ```python
  model.batch_size = 32  # or 16
  model.lstm_hidden_size = 64  # reduce from 128
  ```

## Development Setup

For developers wanting to modify the code:

```bash
# Install development dependencies
pip install jupyter ipykernel black flake8 pytest

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests (when available)
pytest tests/

# Format code
black *.py

# Check code style
flake8 *.py
```

## Next Steps

1. **Start with the Jupyter Notebook**: Run `Dynamic_Data_Labeling_Research.ipynb` cell by cell to understand the system
2. **Experiment with Configuration**: Modify `config.py` to test different parameters
3. **Try Different Stocks**: Change the stock symbols in the configuration
4. **Analyze Results**: Examine the generated visualizations in the `results/` directory
5. **Extend the System**: Add new technical indicators or modify the model architecture

## Support

- Check the comprehensive `README.md` for detailed documentation
- Review the `architecture_diagram.md` for system overview
- Examine `example_usage.py` for implementation patterns
- All code is well-commented for educational purposes

## Safety Notice

⚠️ **Important**: This system is for research and educational purposes only. Never use it for real trading without:
- Extensive backtesting on historical data
- Paper trading validation
- Proper risk management systems
- Understanding of financial regulations

Happy researching! 🚀📈

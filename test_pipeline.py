"""
Simple test script for Windows to verify the pipeline works
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        import torch
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Core libraries imported successfully")
        
        from config import load_config
        print("✅ Config module imported")
        
        from data_processor import DataProcessor, FractionalDifferentiation
        print("✅ Data processor imported")
        
        from dynamic_lstm_model import DynamicLabelingLSTM
        print("✅ LSTM model imported")
        
        from labeling_system import MetaLabeler, TradeSimulator
        print("✅ Labeling system imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\nTesting synthetic data generation...")
    try:
        import pandas as pd
        import numpy as np
        from data_processor import DataProcessor
        from config import load_config
        
        # Create synthetic data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        synthetic_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        print(f"✅ Created synthetic data: {synthetic_data.shape}")
        
        # Test data processing
        config = load_config()
        processor = DataProcessor(config.data)
        processed = processor.process_symbol("TEST", synthetic_data)
        
        print(f"✅ Processed data shape: {processed['features'].shape}")
        print(f"✅ Number of features: {len(processed['feature_names'])}")
        
        return True
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

def test_model():
    """Test model initialization"""
    print("\nTesting model initialization...")
    try:
        import torch
        from dynamic_lstm_model import DynamicLabelingLSTM
        from config import load_config
        
        config = load_config()
        model = DynamicLabelingLSTM(config.model, input_dim=50)
        
        # Test forward pass
        test_input = torch.randn(4, 60, 50)  # batch=4, seq=60, features=50
        output = model(test_input)
        
        print(f"✅ Model created successfully")
        print(f"✅ Forward pass successful")
        print(f"✅ Output shapes: PT={output['pt_raw'].shape}, SL={output['sl_raw'].shape}, TH={output['th_raw'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DYNAMIC LABELING SYSTEM - WINDOWS COMPATIBILITY TEST")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    tests = [
        test_imports,
        test_synthetic_data,
        test_model
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("🎉 ALL TESTS PASSED! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main_pipeline.py --stages data")
        print("2. Or use the Jupyter notebook: jupyter notebook Dynamic_Data_Labeling_Research.ipynb")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTry installing missing dependencies:")
        print("pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

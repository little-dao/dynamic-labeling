"""
Example Usage Script for Dynamic Data Labeling System
Demonstrates various ways to use the system
"""

import os
import numpy as np
import torch
from datetime import datetime

# Import our modules
from config import load_config, Config, DataConfig, ModelConfig
from main_pipeline import DynamicLabelingPipeline
from data_processor import DataProcessor
from evaluation_metrics import FinancialMetricsCalculator, VisualizationTools


def example_1_basic_usage():
    """
    Example 1: Basic usage with default configuration
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Load default configuration
    config = load_config()
    
    # Initialize pipeline
    pipeline = DynamicLabelingPipeline(config)
    
    # Run complete pipeline
    print("Running complete pipeline...")
    results = pipeline.run_complete_pipeline()
    
    # Print results
    if 'evaluate' in results:
        metrics = results['evaluate']['performance_metrics']
        print(f"\nResults:")
        print(f"Total Return: {metrics['total_return']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Win Rate: {metrics['win_rate']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
    
    return results


def example_2_custom_configuration():
    """
    Example 2: Custom configuration for specific stocks
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Custom data configuration - focus on tech stocks
    data_config = DataConfig(
        stock_symbols=["GOOGL", "MSFT", "NVDA", "AMD"],
        crypto_symbols=["BTC-USD"],  # Add one crypto
        start_date="2022-01-01",
        end_date="2024-01-01",
        lookback_window=30,  # Shorter lookback
        technical_indicators=["RSI", "MACD", "BB", "SMA", "EMA"]  # Selected indicators
    )
    
    # Custom model configuration - larger model
    model_config = ModelConfig(
        lstm_hidden_size=256,
        lstm_num_layers=2,
        learning_rate=0.0005,
        batch_size=32,
        num_epochs=50,
        use_attention=True
    )
    
    # Create custom config
    config = Config(data=data_config, model=model_config)
    
    # Initialize pipeline with custom config
    pipeline = DynamicLabelingPipeline(config)
    
    # Run only data processing and label generation
    print("Running data processing and label generation...")
    results = pipeline.run_complete_pipeline(stages=['data', 'labels'])
    
    print(f"\nCustom Configuration Results:")
    print(f"Symbols processed: {results['data']['n_symbols']}")
    print(f"Total sequences: {results['data']['total_sequences']}")
    print(f"Feature dimension: {results['data']['feature_dim']}")
    print(f"Label win rate: {results['labels']['win_rate']:.3f}")
    
    return results


def example_3_step_by_step():
    """
    Example 3: Step-by-step execution with intermediate analysis
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("=" * 60)
    
    config = load_config()
    pipeline = DynamicLabelingPipeline(config)
    
    # Step 1: Data processing
    print("Step 1: Processing data...")
    data_results = pipeline.run_complete_pipeline(stages=['data'])
    print(f"Processed {data_results['data']['n_symbols']} symbols")
    
    # Step 2: Label generation
    print("\nStep 2: Generating labels...")
    label_results = pipeline.run_complete_pipeline(stages=['labels'])
    print(f"Generated {label_results['labels']['n_labels']} labels")
    print(f"Average quality score: {label_results['labels']['avg_quality']:.3f}")
    
    # Step 3: Model initialization
    print("\nStep 3: Initializing model...")
    model_results = pipeline.run_complete_pipeline(stages=['model'])
    print(f"Model initialized with {model_results['model']['total_params']:,} parameters")
    
    # Save pipeline state
    state_path = "pipeline_state_example.pkl"
    pipeline.save_pipeline_state(state_path)
    print(f"Pipeline state saved to {state_path}")
    
    return pipeline


def example_4_evaluation_only():
    """
    Example 4: Evaluation and visualization only
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Evaluation and Visualization")
    print("=" * 60)
    
    # Initialize components
    config = load_config()
    metrics_calculator = FinancialMetricsCalculator(config.evaluation.risk_free_rate)
    visualizer = VisualizationTools()
    
    # Generate sample trade data for demonstration
    print("Generating sample trade data...")
    from labeling_system import Trade, TradeOutcome
    import pandas as pd
    
    np.random.seed(42)
    sample_trades = []
    
    for i in range(200):
        # Simulate various trade outcomes
        outcome_prob = np.random.random()
        if outcome_prob < 0.4:  # 40% profit target hits
            outcome = TradeOutcome.PROFIT_TARGET_HIT
            return_pct = np.random.uniform(0.02, 0.08)
        elif outcome_prob < 0.6:  # 20% stop loss hits
            outcome = TradeOutcome.STOP_LOSS_HIT
            return_pct = np.random.uniform(-0.05, -0.01)
        elif outcome_prob < 0.8:  # 20% time expired profit
            outcome = TradeOutcome.TIME_EXPIRED_PROFIT
            return_pct = np.random.uniform(0.001, 0.025)
        else:  # 20% time expired loss
            outcome = TradeOutcome.TIME_EXPIRED_LOSS
            return_pct = np.random.uniform(-0.025, -0.001)
        
        trade = Trade(
            entry_price=100.0,
            entry_time=pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            profit_target=0.05,
            stop_loss=0.02,
            time_horizon=np.random.randint(5, 20)
        )
        trade.outcome = outcome
        trade.return_pct = return_pct
        trade.duration = np.random.randint(1, trade.time_horizon)
        
        sample_trades.append(trade)
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    performance_metrics = metrics_calculator.calculate_comprehensive_metrics(sample_trades)
    
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {performance_metrics.total_return:.4f}")
    print(f"Sharpe Ratio: {performance_metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {performance_metrics.max_drawdown:.4f}")
    print(f"Win Rate: {performance_metrics.win_rate:.4f}")
    print(f"Profit Factor: {performance_metrics.profit_factor:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Performance comparison (with dummy baseline)
    comparison_data = pd.DataFrame([
        {'Method': 'Dynamic Labeling', **performance_metrics.to_dict()},
        {'Method': 'Buy & Hold', 
         'total_return': 0.12, 'sharpe_ratio': 0.8, 'max_drawdown': -0.25, 
         'win_rate': 0.6, 'volatility': 0.22, 'profit_factor': 1.2}
    ])
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    perf_fig = visualizer.plot_performance_comparison(
        comparison_data, 
        os.path.join(results_dir, 'example_performance_comparison.html')
    )
    
    trade_fig = visualizer.plot_trade_analysis(
        sample_trades,
        os.path.join(results_dir, 'example_trade_analysis.html')
    )
    
    print(f"Visualizations saved to {results_dir}/")
    
    return performance_metrics


def example_5_data_processing_deep_dive():
    """
    Example 5: Deep dive into data processing features
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Data Processing Deep Dive")
    print("=" * 60)
    
    config = load_config()
    processor = DataProcessor(config.data)
    
    # Download data for a single symbol
    print("Downloading data for AAPL...")
    raw_data = processor.download_data(["AAPL"], "2023-01-01", "2024-01-01")
    
    if "AAPL" in raw_data:
        df = raw_data["AAPL"]
        print(f"Downloaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Process the symbol
        print("\nProcessing AAPL data...")
        processed = processor.process_symbol("AAPL", df)
        
        print(f"Feature matrix shape: {processed['features'].shape}")
        print(f"Number of features: {len(processed['feature_names'])}")
        print(f"Feature names (first 10): {processed['feature_names'][:10]}")
        
        # Show some fractional differentiation analysis
        print("\nFractional differentiation analysis:")
        from data_processor import FractionalDifferentiation
        
        ffd = FractionalDifferentiation()
        close_prices = df['Close']
        
        # Find optimal d value
        d_opt = ffd.find_min_ffd_order(close_prices)
        print(f"Optimal fractional differentiation order: {d_opt:.3f}")
        
        # Apply FFD
        if d_opt > 0:
            ffd_series = ffd.fracDiff_FFD(close_prices, d_opt)
            print(f"FFD series length: {len(ffd_series)}")
            print(f"Original series std: {close_prices.std():.4f}")
            print(f"FFD series std: {ffd_series.std():.4f}")
    
    return processed if "AAPL" in raw_data else None


def main():
    """
    Main function to run all examples
    """
    print("Dynamic Data Labeling System - Example Usage")
    print("=" * 60)
    
    # Set up environment
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_custom_configuration()
        example_3_step_by_step()
        example_4_evaluation_only()
        example_5_data_processing_deep_dive()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the 'results/' directory for generated visualizations.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("This is normal if you don't have all dependencies installed.")
        print("Install requirements with: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

"""
Main Pipeline for Dynamic Data Labeling System
Orchestrates the complete workflow from data processing to model evaluation
"""

import os
import pickle
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

# Import our modules
from config import load_config, Config
from data_processor import DataProcessor
from dynamic_lstm_model import DynamicLabelingLSTM, ModelTrainer, DynamicLabelingLoss
from labeling_system import AdaptiveLabelGenerator, FeedbackLoop
from evaluation_metrics import FinancialMetricsCalculator, RobustnessAnalyzer, VisualizationTools


# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DynamicLabelingDataset(data_utils.Dataset):
    """
    PyTorch Dataset for dynamic labeling
    """
    
    def __init__(self, features: np.ndarray, labels: Dict[str, np.ndarray]):
        self.features = torch.FloatTensor(features)
        self.pt_labels = torch.FloatTensor(labels.get('pt', np.zeros(len(features))))
        self.sl_labels = torch.FloatTensor(labels.get('sl', np.zeros(len(features))))
        self.th_labels = torch.FloatTensor(labels.get('th', np.zeros(len(features))))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'pt': self.pt_labels[idx],
            'sl': self.sl_labels[idx],
            'th': self.th_labels[idx]
        }


class DynamicLabelingPipeline:
    """
    Main pipeline class for the dynamic labeling system
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.data_processor = DataProcessor(config.data)
        self.label_generator = AdaptiveLabelGenerator(config.trading)
        self.metrics_calculator = FinancialMetricsCalculator(config.evaluation.risk_free_rate)
        self.robustness_analyzer = RobustnessAnalyzer(config.evaluation)
        self.visualizer = VisualizationTools()
        
        # Model and trainer (initialized later)
        self.model = None
        self.trainer = None
        self.feedback_loop = None
        
        # Data storage
        self.processed_data = None
        self.features = None
        self.labels = None
        
    def run_complete_pipeline(self, stages: List[str] = None) -> Dict:
        """
        Run the complete pipeline
        
        Args:
            stages: List of stages to run. If None, runs all stages.
                   Options: ['data', 'labels', 'model', 'train', 'evaluate']
        
        Returns:
            Dictionary with results from each stage
        """
        if stages is None:
            stages = ['data', 'labels', 'model', 'train', 'evaluate']
        
        results = {}
        
        try:
            if 'data' in stages:
                logger.info("Stage 1: Data Processing")
                results['data'] = self.stage_data_processing()
            
            if 'labels' in stages:
                logger.info("Stage 2: Label Generation")
                results['labels'] = self.stage_label_generation()
            
            if 'model' in stages:
                logger.info("Stage 3: Model Initialization")
                results['model'] = self.stage_model_initialization()
            
            if 'train' in stages:
                logger.info("Stage 4: Model Training")
                results['train'] = self.stage_model_training()
            
            if 'evaluate' in stages:
                logger.info("Stage 5: Evaluation and Analysis")
                results['evaluate'] = self.stage_evaluation()
            
            logger.info("Pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def stage_data_processing(self) -> Dict:
        """
        Stage 1: Process raw data into features
        """
        try:
            # Check if processed data already exists
            data_path = os.path.join(self.config.data_dir, "processed_data.pkl")
            raw_data_path = os.path.join(self.config.data_dir, "raw_data.pkl")
            
            if os.path.exists(data_path):
                logger.info("Loading existing processed data...")
                with open(data_path, "rb") as f:
                    self.processed_data = pickle.load(f)
            elif os.path.exists(raw_data_path):
                logger.info("Found raw data, processing...")
                # Load raw data and process it
                with open(raw_data_path, "rb") as f:
                    raw_data = pickle.load(f)
                
                self.processed_data = {}
                for symbol, df in raw_data.items():
                    try:
                        processed = self.data_processor.process_symbol(symbol, df)
                        self.processed_data[symbol] = processed
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                
                # Save processed data
                with open(data_path, "wb") as f:
                    pickle.dump(self.processed_data, f)
                logger.info(f"Processed data saved to {data_path}")
            else:
                logger.info("No existing data found, downloading...")
                self.processed_data = self.data_processor.process_all_data()
                
                # Save processed data
                with open(data_path, "wb") as f:
                    pickle.dump(self.processed_data, f)
                logger.info(f"Processed data saved to {data_path}")
            
            # Extract features
            all_features = []
            all_indices = []
            symbol_info = []
            
            for symbol, data in self.processed_data.items():
                features = data['features']
                indices = data['indices']
                
                all_features.append(features)
                all_indices.extend([(symbol, idx) for idx in indices])
                symbol_info.extend([symbol] * len(features))
                
                logger.info(f"{symbol}: {features.shape[0]} sequences, {features.shape[1]} timesteps, {features.shape[2]} features")
            
            # Combine all features
            self.features = np.vstack(all_features)
            self.symbol_info = symbol_info
            self.indices_info = all_indices
            
            logger.info(f"Total features shape: {self.features.shape}")
            
            return {
                'n_symbols': len(self.processed_data),
                'total_sequences': len(self.features),
                'feature_dim': self.features.shape[2],
                'sequence_length': self.features.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise
    
    def stage_label_generation(self) -> Dict:
        """
        Stage 2: Generate initial labels using adaptive labeling
        """
        try:
            # Create dummy predictions for initial label generation
            n_samples = len(self.features)
            
            # Initialize with conservative predictions
            dummy_predictions = {}
            price_data = {}
            prediction_times = {}
            
            for symbol, data in self.processed_data.items():
                symbol_indices = [i for i, (s, _) in enumerate(self.indices_info) if s == symbol]
                n_symbol_samples = len(symbol_indices)
                
                if n_symbol_samples > 0:
                    dummy_predictions[symbol] = {
                        'profit_taking': np.random.uniform(
                            self.config.trading.pt_min, 
                            self.config.trading.pt_max, 
                            n_symbol_samples
                        ),
                        'stop_loss': np.random.uniform(
                            self.config.trading.sl_min, 
                            self.config.trading.sl_max, 
                            n_symbol_samples
                        ),
                        'time_horizon': np.random.randint(
                            self.config.trading.th_min, 
                            self.config.trading.th_max, 
                            n_symbol_samples
                        )
                    }
                    
                    # Prepare price data
                    ohlcv_data = data['processed_data'][['Open', 'High', 'Low', 'Close', 'Volume']]
                    price_data[symbol] = ohlcv_data
                    
                    # Get prediction times for this symbol
                    symbol_times = [idx for s, idx in self.indices_info if s == symbol]
                    prediction_times[symbol] = np.array(symbol_times)
            
            # Generate labels
            label_results = self.label_generator.generate_initial_labels(
                dummy_predictions, price_data, prediction_times
            )
            
            # Store labels for model training
            self.labels = {
                'pt': label_results['continuous'],  # Use continuous for PT
                'sl': label_results['continuous'],  # Use continuous for SL  
                'th': label_results['continuous']   # Use continuous for TH
            }
            
            logger.info(f"Generated {len(label_results['binary'])} labels")
            logger.info(f"Label distribution - Binary: {np.mean(label_results['binary']):.3f}")
            logger.info(f"Label distribution - Continuous: {np.mean(label_results['continuous']):.3f}")
            
            return {
                'n_labels': len(label_results['binary']),
                'win_rate': np.mean(label_results['binary']),
                'avg_quality': np.mean(label_results['continuous']),
                'label_types': list(label_results.keys())
            }
            
        except Exception as e:
            logger.error(f"Label generation failed: {str(e)}")
            raise
    
    def stage_model_initialization(self) -> Dict:
        """
        Stage 3: Initialize and setup the LSTM model
        """
        try:
            # Get input dimensions
            input_dim = self.features.shape[2]
            
            # Initialize model
            self.model = DynamicLabelingLSTM(self.config.model, input_dim)
            
            # Initialize trainer
            self.trainer = ModelTrainer(self.model, self.config.model, str(self.device))
            
            # Initialize feedback loop
            self.feedback_loop = FeedbackLoop(self.model, self.config.trading)
            
            # Model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model initialized with {total_params:,} total parameters")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return {
                'input_dim': input_dim,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_architecture': str(self.model)
            }
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def stage_model_training(self) -> Dict:
        """
        Stage 4: Train the model with cross-validation
        """
        try:
            # Prepare training data
            train_features, val_features, train_labels, val_labels = self._prepare_training_data()
            
            # Create data loaders
            train_dataset = DynamicLabelingDataset(train_features, train_labels)
            val_dataset = DynamicLabelingDataset(val_features, val_labels)
            
            train_loader = data_utils.DataLoader(
                train_dataset, 
                batch_size=self.config.model.batch_size, 
                shuffle=True,
                num_workers=self.config.num_workers
            )
            
            val_loader = data_utils.DataLoader(
                val_dataset, 
                batch_size=self.config.model.batch_size, 
                shuffle=False,
                num_workers=self.config.num_workers
            )
            
            # Train model
            training_history = self.trainer.train(train_loader, val_loader)
            
            # Save trained model
            model_path = os.path.join(self.config.models_dir, "dynamic_labeling_model.pth")
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Training summary
            final_train_loss = training_history['train_losses'][-1]['total_loss']
            final_val_loss = training_history['val_losses'][-1]['total_loss']
            best_val_loss = self.trainer.best_val_loss
            
            logger.info(f"Training completed!")
            logger.info(f"Final train loss: {final_train_loss:.6f}")
            logger.info(f"Final val loss: {final_val_loss:.6f}")
            logger.info(f"Best val loss: {best_val_loss:.6f}")
            
            return {
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'best_val_loss': best_val_loss,
                'training_epochs': len(training_history['train_losses']),
                'training_history': training_history
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def stage_evaluation(self) -> Dict:
        """
        Stage 5: Comprehensive evaluation and analysis
        """
        try:
            # Generate predictions on test data
            test_features, test_labels = self._prepare_test_data()
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                test_features_tensor = torch.FloatTensor(test_features).to(self.device)
                predictions = self.model.predict(test_features_tensor)
            
            # Convert predictions to numpy
            pred_dict = {}
            test_symbols = list(set([s for s, _ in self.indices_info[-len(test_features):]]))
            
            for symbol in test_symbols:
                symbol_indices = [i for i, (s, _) in enumerate(self.indices_info[-len(test_features):]) if s == symbol]
                if symbol_indices:
                    pred_dict[symbol] = {
                        'profit_taking': predictions['profit_taking'][symbol_indices].cpu().numpy(),
                        'stop_loss': predictions['stop_loss'][symbol_indices].cpu().numpy(),
                        'time_horizon': predictions['time_horizon'][symbol_indices].cpu().numpy()
                    }
            
            # Prepare data for evaluation
            price_data = {}
            prediction_times = {}
            
            for symbol in test_symbols:
                if symbol in self.processed_data:
                    ohlcv_data = self.processed_data[symbol]['processed_data'][['Open', 'High', 'Low', 'Close', 'Volume']]
                    price_data[symbol] = ohlcv_data
                    
                    symbol_times = [idx for s, idx in self.indices_info[-len(test_features):] if s == symbol]
                    prediction_times[symbol] = np.array(symbol_times)
            
            # Generate evaluation labels
            eval_results = self.label_generator.generate_initial_labels(
                pred_dict, price_data, prediction_times
            )
            
            # Calculate performance metrics
            trades = eval_results['trades']
            performance_metrics = self.metrics_calculator.calculate_comprehensive_metrics(trades)
            
            # Robustness analysis
            original_labels = eval_results['binary']
            robustness_results = self.robustness_analyzer.evaluate_labeling_robustness(
                trades, original_labels
            )
            
            # Create visualizations
            comparison_data = pd.DataFrame([{
                'Method': 'Dynamic Labeling',
                **performance_metrics.to_dict()
            }])
            
            # Save visualizations
            results_dir = self.config.results_dir
            
            perf_fig = self.visualizer.plot_performance_comparison(
                comparison_data, 
                os.path.join(results_dir, 'performance_comparison.html')
            )
            
            robustness_fig = self.visualizer.plot_robustness_analysis(
                robustness_results,
                os.path.join(results_dir, 'robustness_analysis.html')
            )
            
            trade_fig = self.visualizer.plot_trade_analysis(
                trades,
                os.path.join(results_dir, 'trade_analysis.html')
            )
            
            # Save evaluation results
            eval_summary = {
                'performance_metrics': performance_metrics.to_dict(),
                'robustness_results': robustness_results,
                'n_trades': len(trades),
                'test_symbols': test_symbols
            }
            
            results_path = os.path.join(results_dir, 'evaluation_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(eval_summary, f)
            
            logger.info("Evaluation completed!")
            logger.info(f"Performance Metrics:")
            logger.info(f"  Total Return: {performance_metrics.total_return:.4f}")
            logger.info(f"  Sharpe Ratio: {performance_metrics.sharpe_ratio:.4f}")
            logger.info(f"  Max Drawdown: {performance_metrics.max_drawdown:.4f}")
            logger.info(f"  Win Rate: {performance_metrics.win_rate:.4f}")
            logger.info(f"Robustness Score: {robustness_results['robustness_score']:.4f}")
            
            return eval_summary
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Prepare training and validation data"""
        # Time-based split to avoid look-ahead bias
        split_idx = int(len(self.features) * (1 - self.config.evaluation.test_size))
        
        train_features = self.features[:split_idx]
        val_features = self.features[split_idx:]
        
        train_labels = {
            'pt': self.labels['pt'][:split_idx],
            'sl': self.labels['sl'][:split_idx],
            'th': self.labels['th'][:split_idx]
        }
        
        val_labels = {
            'pt': self.labels['pt'][split_idx:],
            'sl': self.labels['sl'][split_idx:],
            'th': self.labels['th'][split_idx:]
        }
        
        return train_features, val_features, train_labels, val_labels
    
    def _prepare_test_data(self) -> Tuple[np.ndarray, Dict]:
        """Prepare test data for evaluation"""
        # Use the last portion of data as test set
        test_size = int(len(self.features) * 0.2)
        test_features = self.features[-test_size:]
        
        test_labels = {
            'pt': self.labels['pt'][-test_size:],
            'sl': self.labels['sl'][-test_size:],
            'th': self.labels['th'][-test_size:]
        }
        
        return test_features, test_labels
    
    def save_pipeline_state(self, filepath: str):
        """Save the current pipeline state"""
        state = {
            'config': self.config,
            'processed_data': self.processed_data,
            'features': self.features,
            'labels': self.labels,
            'symbol_info': self.symbol_info,
            'indices_info': self.indices_info
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Pipeline state saved to {filepath}")
    
    def load_pipeline_state(self, filepath: str):
        """Load a saved pipeline state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.processed_data = state['processed_data']
        self.features = state['features']
        self.labels = state['labels']
        self.symbol_info = state['symbol_info']
        self.indices_info = state['indices_info']
        
        logger.info(f"Pipeline state loaded from {filepath}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Dynamic Data Labeling Pipeline")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stages', nargs='+', 
                       choices=['data', 'labels', 'model', 'train', 'evaluate'],
                       default=['data', 'labels', 'model', 'train', 'evaluate'],
                       help='Pipeline stages to run')
    parser.add_argument('--save-state', type=str, help='Path to save pipeline state')
    parser.add_argument('--load-state', type=str, help='Path to load pipeline state')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Initialize pipeline
    pipeline = DynamicLabelingPipeline(config)
    
    # Load state if specified
    if args.load_state:
        pipeline.load_pipeline_state(args.load_state)
    
    try:
        # Run pipeline
        results = pipeline.run_complete_pipeline(args.stages)
        
        # Save state if specified
        if args.save_state:
            pipeline.save_pipeline_state(args.save_state)
        
        logger.info("Pipeline execution completed successfully!")
        
        # Print final summary
        if 'evaluate' in results:
            eval_results = results['evaluate']
            print("\n" + "="*50)
            print("FINAL RESULTS SUMMARY")
            print("="*50)
            
            perf_metrics = eval_results['performance_metrics']
            print(f"Total Return: {perf_metrics['total_return']:.4f}")
            print(f"Sharpe Ratio: {perf_metrics['sharpe_ratio']:.4f}")
            print(f"Max Drawdown: {perf_metrics['max_drawdown']:.4f}")
            print(f"Win Rate: {perf_metrics['win_rate']:.4f}")
            print(f"Number of Trades: {eval_results['n_trades']}")
            print(f"Robustness Score: {eval_results['robustness_results']['robustness_score']:.4f}")
            print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

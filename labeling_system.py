"""
Dynamic Labeling and Meta-Labeling System
Implements trade simulation, outcome determination, and adaptive feedback loops
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from config import TradingConfig


class TradeOutcome(Enum):
    """Enumeration for trade outcomes"""
    PROFIT_TARGET_HIT = "profit_target"
    STOP_LOSS_HIT = "stop_loss"
    TIME_EXPIRED_PROFIT = "time_expired_profit"
    TIME_EXPIRED_LOSS = "time_expired_loss"
    ONGOING = "ongoing"


@dataclass
class Trade:
    """Individual trade data structure"""
    entry_price: float
    entry_time: pd.Timestamp
    profit_target: float
    stop_loss: float
    time_horizon: int
    
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    outcome: Optional[TradeOutcome] = None
    return_pct: Optional[float] = None
    duration: Optional[int] = None
    
    def __post_init__(self):
        self.profit_target_price = self.entry_price * (1 + self.profit_target)
        self.stop_loss_price = self.entry_price * (1 - self.stop_loss)


class TradeSimulator:
    """
    Simulates trades based on predicted parameters and determines outcomes
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        
    def simulate_single_trade(self, trade: Trade, price_data: pd.DataFrame) -> Trade:
        """
        Simulate a single trade and determine its outcome
        
        Args:
            trade: Trade object with entry parameters
            price_data: OHLCV data starting from entry time
            
        Returns:
            Updated trade with outcome information
        """
        # Get price data from entry time onwards
        entry_idx = price_data.index.get_loc(trade.entry_time)
        max_end_idx = min(entry_idx + trade.time_horizon + 1, len(price_data))
        
        trade_period_data = price_data.iloc[entry_idx:max_end_idx]
        
        for i, (timestamp, row) in enumerate(trade_period_data.iterrows()):
            if i == 0:  # Entry day
                continue
                
            current_high = row['High']
            current_low = row['Low']
            current_close = row['Close']
            
            # Check if profit target hit
            if current_high >= trade.profit_target_price:
                trade.exit_price = trade.profit_target_price
                trade.exit_time = timestamp
                trade.outcome = TradeOutcome.PROFIT_TARGET_HIT
                trade.return_pct = trade.profit_target
                trade.duration = i
                break
            
            # Check if stop loss hit
            if current_low <= trade.stop_loss_price:
                trade.exit_price = trade.stop_loss_price
                trade.exit_time = timestamp
                trade.outcome = TradeOutcome.STOP_LOSS_HIT
                trade.return_pct = -trade.stop_loss
                trade.duration = i
                break
            
            # Check if time horizon reached
            if i == len(trade_period_data) - 1:
                trade.exit_price = current_close
                trade.exit_time = timestamp
                trade.return_pct = (current_close - trade.entry_price) / trade.entry_price
                trade.duration = i
                
                if trade.return_pct > 0:
                    trade.outcome = TradeOutcome.TIME_EXPIRED_PROFIT
                else:
                    trade.outcome = TradeOutcome.TIME_EXPIRED_LOSS
                break
        
        return trade
    
    def simulate_portfolio_trades(self, predictions: Dict[str, np.ndarray], 
                                price_data: Dict[str, pd.DataFrame],
                                prediction_times: Dict[str, np.ndarray]) -> List[Trade]:
        """
        Simulate trades for multiple symbols based on model predictions
        
        Args:
            predictions: Model predictions for each symbol
            price_data: Price data for each symbol
            prediction_times: Timestamps for predictions
            
        Returns:
            List of simulated trades
        """
        all_trades = []
        
        for symbol in predictions:
            symbol_predictions = predictions[symbol]
            symbol_prices = price_data[symbol]
            symbol_times = prediction_times[symbol]
            
            for i, pred_time in enumerate(symbol_times):
                if pred_time not in symbol_prices.index:
                    continue
                
                # Create trade from predictions
                entry_price = symbol_prices.loc[pred_time, 'Close']
                pt_threshold = float(symbol_predictions['profit_taking'][i])
                sl_threshold = float(symbol_predictions['stop_loss'][i])
                time_horizon = int(symbol_predictions['time_horizon'][i])
                
                trade = Trade(
                    entry_price=entry_price,
                    entry_time=pred_time,
                    profit_target=pt_threshold,
                    stop_loss=sl_threshold,
                    time_horizon=time_horizon
                )
                
                # Simulate trade
                simulated_trade = self.simulate_single_trade(trade, symbol_prices)
                all_trades.append(simulated_trade)
        
        return all_trades


class MetaLabeler:
    """
    Generates meta-labels based on trade outcomes for adaptive learning
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        
    def generate_binary_labels(self, trades: List[Trade]) -> np.ndarray:
        """
        Generate binary good/bad labels from trade outcomes
        
        Args:
            trades: List of simulated trades
            
        Returns:
            Binary labels (1 for good, 0 for bad)
        """
        labels = []
        
        for trade in trades:
            if trade.outcome == TradeOutcome.PROFIT_TARGET_HIT:
                labels.append(1)  # Good trade
            elif trade.outcome == TradeOutcome.STOP_LOSS_HIT:
                labels.append(0)  # Bad trade
            elif trade.outcome == TradeOutcome.TIME_EXPIRED_PROFIT:
                # Consider as good if return > transaction cost
                if trade.return_pct > self.config.transaction_cost:
                    labels.append(1)
                else:
                    labels.append(0)
            else:  # TIME_EXPIRED_LOSS
                labels.append(0)  # Bad trade
        
        return np.array(labels)
    
    def generate_continuous_labels(self, trades: List[Trade]) -> np.ndarray:
        """
        Generate continuous quality labels based on return/risk metrics
        
        Args:
            trades: List of simulated trades
            
        Returns:
            Continuous quality scores
        """
        labels = []
        
        for trade in trades:
            if trade.return_pct is None:
                labels.append(0.0)
                continue
            
            # Risk-adjusted return score
            risk_adjustment = min(trade.stop_loss, 0.05)  # Cap risk adjustment
            time_adjustment = max(0.1, 1.0 - trade.duration / 30.0)  # Prefer shorter duration
            
            # Base score from return
            base_score = trade.return_pct / risk_adjustment
            
            # Apply time adjustment
            quality_score = base_score * time_adjustment
            
            # Normalize to [0, 1]
            quality_score = 1.0 / (1.0 + np.exp(-quality_score))
            
            labels.append(quality_score)
        
        return np.array(labels)
    
    def generate_multi_class_labels(self, trades: List[Trade]) -> np.ndarray:
        """
        Generate multi-class labels for more granular feedback
        
        Args:
            trades: List of simulated trades
            
        Returns:
            Multi-class labels (0: bad, 1: neutral, 2: good)
        """
        labels = []
        
        for trade in trades:
            if trade.outcome == TradeOutcome.PROFIT_TARGET_HIT:
                labels.append(2)  # Excellent
            elif trade.outcome == TradeOutcome.TIME_EXPIRED_PROFIT:
                if trade.return_pct > 2 * self.config.transaction_cost:
                    labels.append(2)  # Good
                else:
                    labels.append(1)  # Neutral
            elif trade.outcome == TradeOutcome.TIME_EXPIRED_LOSS:
                if abs(trade.return_pct) < self.config.transaction_cost:
                    labels.append(1)  # Neutral
                else:
                    labels.append(0)  # Bad
            else:  # STOP_LOSS_HIT
                labels.append(0)  # Bad
        
        return np.array(labels)


class AdaptiveLabelGenerator:
    """
    Adaptive label generation system that learns from trade outcomes
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trade_simulator = TradeSimulator(config)
        self.meta_labeler = MetaLabeler(config)
        
        # Track performance metrics
        self.trade_history = []
        self.performance_metrics = {}
        
    def generate_initial_labels(self, predictions: Dict[str, np.ndarray],
                              price_data: Dict[str, pd.DataFrame],
                              prediction_times: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Generate initial labels from model predictions
        
        Args:
            predictions: Model predictions
            price_data: Historical price data
            prediction_times: Timestamps for predictions
            
        Returns:
            Dictionary of generated labels
        """
        # Simulate trades
        trades = self.trade_simulator.simulate_portfolio_trades(
            predictions, price_data, prediction_times
        )
        
        # Store trade history
        self.trade_history.extend(trades)
        
        # Generate different types of labels
        binary_labels = self.meta_labeler.generate_binary_labels(trades)
        continuous_labels = self.meta_labeler.generate_continuous_labels(trades)
        multi_class_labels = self.meta_labeler.generate_multi_class_labels(trades)
        
        # Calculate performance metrics
        self._update_performance_metrics(trades)
        
        return {
            'binary': binary_labels,
            'continuous': continuous_labels,
            'multi_class': multi_class_labels,
            'trades': trades
        }
    
    def update_labels_with_feedback(self, new_predictions: Dict[str, np.ndarray],
                                  new_price_data: Dict[str, pd.DataFrame],
                                  new_prediction_times: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update labels with new feedback for adaptive learning
        
        Args:
            new_predictions: New model predictions
            new_price_data: New price data
            new_prediction_times: New prediction timestamps
            
        Returns:
            Updated labels incorporating feedback
        """
        # Generate new labels
        new_labels = self.generate_initial_labels(
            new_predictions, new_price_data, new_prediction_times
        )
        
        # Adaptive adjustments based on recent performance
        if len(self.trade_history) > 100:  # Minimum trades for adaptation
            recent_performance = self._analyze_recent_performance()
            new_labels = self._adapt_labels_based_on_performance(new_labels, recent_performance)
        
        return new_labels
    
    def _update_performance_metrics(self, trades: List[Trade]):
        """Update performance tracking metrics"""
        if not trades:
            return
        
        returns = [t.return_pct for t in trades if t.return_pct is not None]
        durations = [t.duration for t in trades if t.duration is not None]
        
        if returns:
            self.performance_metrics.update({
                'total_trades': len(trades),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'avg_return': np.mean(returns),
                'avg_duration': np.mean(durations) if durations else 0,
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns)
            })
    
    def _analyze_recent_performance(self, lookback: int = 50) -> Dict[str, float]:
        """Analyze recent trading performance for adaptation"""
        recent_trades = self.trade_history[-lookback:]
        recent_returns = [t.return_pct for t in recent_trades if t.return_pct is not None]
        
        if not recent_returns:
            return {}
        
        return {
            'recent_win_rate': len([r for r in recent_returns if r > 0]) / len(recent_returns),
            'recent_avg_return': np.mean(recent_returns),
            'recent_volatility': np.std(recent_returns),
            'trend': np.polyfit(range(len(recent_returns)), recent_returns, 1)[0]
        }
    
    def _adapt_labels_based_on_performance(self, labels: Dict[str, np.ndarray], 
                                         performance: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Adapt label generation based on recent performance"""
        adapted_labels = labels.copy()
        
        # If recent performance is poor, be more conservative
        if performance.get('recent_win_rate', 0.5) < 0.4:
            # Make binary labels more conservative
            conservative_threshold = 0.6
            adapted_labels['binary'] = (labels['continuous'] > conservative_threshold).astype(int)
            
            # Reduce continuous scores
            adapted_labels['continuous'] = labels['continuous'] * 0.8
        
        # If recent volatility is high, prefer shorter time horizons
        if performance.get('recent_volatility', 0.02) > 0.05:
            # This would be implemented in the model training phase
            pass
        
        return adapted_labels
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from return series"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get comprehensive performance summary"""
        return self.performance_metrics.copy()
    
    def get_recent_trades(self, n: int = 20) -> List[Trade]:
        """Get most recent trades"""
        return self.trade_history[-n:] if len(self.trade_history) >= n else self.trade_history


class FeedbackLoop:
    """
    Implements the feedback loop for continuous model improvement
    """
    
    def __init__(self, model, config: TradingConfig):
        self.model = model
        self.config = config
        self.label_generator = AdaptiveLabelGenerator(config)
        
        # Track model evolution
        self.model_generations = []
        self.performance_history = []
        
    def generate_training_data(self, features: np.ndarray, 
                             price_data: Dict[str, pd.DataFrame],
                             prediction_times: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate training data with adaptive labels
        
        Args:
            features: Input features for model
            price_data: Price data for simulation
            prediction_times: Timestamps for predictions
            
        Returns:
            Tuple of (features, labels)
        """
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            features_tensor = torch.FloatTensor(features)
            predictions = self.model.predict(features_tensor)
            
            # Convert to numpy and organize by symbol
            pred_dict = {}
            for symbol in price_data.keys():
                pred_dict[symbol] = {
                    'profit_taking': predictions['profit_taking'].numpy(),
                    'stop_loss': predictions['stop_loss'].numpy(),
                    'time_horizon': predictions['time_horizon'].numpy()
                }
        
        # Generate labels
        labels = self.label_generator.generate_initial_labels(
            pred_dict, price_data, prediction_times
        )
        
        return features, labels
    
    def update_model_with_feedback(self, features: np.ndarray, labels: Dict[str, np.ndarray]):
        """
        Update model using feedback labels
        
        Args:
            features: Input features
            labels: Generated labels for training
        """
        # This would involve retraining the model with new labels
        # Implementation depends on the specific training framework
        print(f"Updating model with {len(features)} samples and feedback labels")
        
        # Store performance for tracking
        performance = self.label_generator.get_performance_summary()
        self.performance_history.append(performance)
        
        print(f"Current performance metrics: {performance}")
    
    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained based on performance
        
        Returns:
            True if retraining is recommended
        """
        if len(self.performance_history) < 2:
            return False
        
        current_perf = self.performance_history[-1]
        previous_perf = self.performance_history[-2]
        
        # Retrain if performance degraded significantly
        current_sharpe = current_perf.get('sharpe_ratio', 0)
        previous_sharpe = previous_perf.get('sharpe_ratio', 0)
        
        if current_sharpe < previous_sharpe * 0.8:  # 20% degradation
            return True
        
        # Retrain if win rate dropped significantly
        current_win_rate = current_perf.get('win_rate', 0.5)
        previous_win_rate = previous_perf.get('win_rate', 0.5)
        
        if current_win_rate < previous_win_rate * 0.9:  # 10% degradation
            return True
        
        return False


if __name__ == "__main__":
    # Example usage
    from config import load_config
    import pickle
    
    config = load_config()
    
    # Initialize components
    label_generator = AdaptiveLabelGenerator(config.trading)
    
    # Load some sample data (this would come from your data processor)
    try:
        with open("data/processed_data.pkl", "rb") as f:
            processed_data = pickle.load(f)
        
        # Example of generating labels
        symbol = list(processed_data.keys())[0]
        sample_data = processed_data[symbol]
        
        # Create dummy predictions for demonstration
        n_samples = 100
        dummy_predictions = {
            symbol: {
                'profit_taking': np.random.uniform(0.02, 0.10, n_samples),
                'stop_loss': np.random.uniform(0.01, 0.05, n_samples),
                'time_horizon': np.random.randint(5, 20, n_samples)
            }
        }
        
        price_data = {symbol: sample_data['processed_data'][['Open', 'High', 'Low', 'Close', 'Volume']]}
        pred_times = {symbol: sample_data['indices'][:n_samples]}
        
        # Generate labels
        labels = label_generator.generate_initial_labels(
            dummy_predictions, price_data, pred_times
        )
        
        print("Label generation successful!")
        print(f"Generated {len(labels['binary'])} labels")
        print(f"Win rate: {np.mean(labels['binary']):.3f}")
        print(f"Average quality score: {np.mean(labels['continuous']):.3f}")
        
        # Print performance summary
        performance = label_generator.get_performance_summary()
        print("\nPerformance Summary:")
        for key, value in performance.items():
            print(f"{key}: {value:.4f}")
            
    except FileNotFoundError:
        print("No processed data found. Run data_processor.py first.")

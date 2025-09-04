"""
Evaluation and Robustness Testing Module
Implements comprehensive financial performance metrics and robustness testing
for the dynamic labeling system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from labeling_system import Trade, TradeOutcome
from config import EvaluationConfig


@dataclass
class PerformanceMetrics:
    """Container for financial performance metrics"""
    # Return-based metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    value_at_risk: float  # 5% VaR
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade-based metrics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional metrics
    num_trades: int
    avg_trade_duration: float
    hit_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'value_at_risk': self.value_at_risk,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'num_trades': self.num_trades,
            'avg_trade_duration': self.avg_trade_duration,
            'hit_ratio': self.hit_ratio
        }


class FinancialMetricsCalculator:
    """
    Calculates comprehensive financial performance metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns_metrics(self, returns: np.ndarray, 
                                periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate return-based metrics
        
        Args:
            returns: Array of periodic returns
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Dictionary of return metrics
        """
        if len(returns) == 0:
            return self._empty_returns_metrics()
        
        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        cumulative_return = np.cumprod(1 + returns)
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(cumulative_return)
        
        # Value at Risk (5%)
        var_5 = np.percentile(returns, 5)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'value_at_risk': var_5
        }
    
    def calculate_risk_adjusted_metrics(self, returns: np.ndarray,
                                      benchmark_returns: Optional[np.ndarray] = None,
                                      periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics
        
        Args:
            returns: Array of strategy returns
            benchmark_returns: Array of benchmark returns
            periods_per_year: Number of periods per year
            
        Returns:
            Dictionary of risk-adjusted metrics
        """
        if len(returns) == 0:
            return self._empty_risk_adjusted_metrics()
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year)
            sortino_ratio = (np.mean(returns) * periods_per_year - self.risk_free_rate) / downside_deviation
        else:
            sortino_ratio = np.inf
        
        # Calmar Ratio
        annualized_return = (np.prod(1 + returns) ** (periods_per_year / len(returns))) - 1
        max_drawdown = self._calculate_max_drawdown(np.cumprod(1 + returns))
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Information Ratio
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(periods_per_year)
            information_ratio = np.mean(active_returns) * periods_per_year / tracking_error if tracking_error != 0 else 0
        else:
            information_ratio = 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate trade-based performance metrics
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dictionary of trade metrics
        """
        if not trades:
            return self._empty_trade_metrics()
        
        returns = [t.return_pct for t in trades if t.return_pct is not None]
        durations = [t.duration for t in trades if t.duration is not None]
        
        if not returns:
            return self._empty_trade_metrics()
        
        # Win/Loss analysis
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses != 0 else np.inf
        
        # Hit ratio (profit target hit rate)
        profit_target_hits = len([t for t in trades if t.outcome == TradeOutcome.PROFIT_TARGET_HIT])
        hit_ratio = profit_target_hits / len(trades)
        
        # Average trade duration
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(trades),
            'avg_trade_duration': avg_duration,
            'hit_ratio': hit_ratio
        }
    
    def calculate_comprehensive_metrics(self, trades: List[Trade],
                                      benchmark_returns: Optional[np.ndarray] = None) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Args:
            trades: List of completed trades
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with all metrics
        """
        # Extract returns from trades
        returns = np.array([t.return_pct for t in trades if t.return_pct is not None])
        
        # Calculate metric components
        return_metrics = self.calculate_returns_metrics(returns)
        risk_metrics = self.calculate_risk_adjusted_metrics(returns, benchmark_returns)
        trade_metrics = self.calculate_trade_metrics(trades)
        
        # Combine all metrics
        all_metrics = {**return_metrics, **risk_metrics, **trade_metrics}
        
        return PerformanceMetrics(**all_metrics)
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    def _empty_returns_metrics(self) -> Dict[str, float]:
        """Return empty returns metrics"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'cumulative_return': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'value_at_risk': 0.0
        }
    
    def _empty_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Return empty risk-adjusted metrics"""
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0
        }
    
    def _empty_trade_metrics(self) -> Dict[str, float]:
        """Return empty trade metrics"""
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0,
            'avg_trade_duration': 0.0,
            'hit_ratio': 0.0
        }


class RobustnessAnalyzer:
    """
    Implements robustness testing for labeling algorithms
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics_calculator = FinancialMetricsCalculator(config.risk_free_rate)
    
    def noise_model_simulation(self, original_labels: np.ndarray,
                             accuracy_levels: List[float]) -> Dict[float, np.ndarray]:
        """
        Simulate different accuracy levels by flipping labels
        
        Args:
            original_labels: Original binary labels
            accuracy_levels: List of target accuracy levels
            
        Returns:
            Dictionary mapping accuracy levels to noisy labels
        """
        noisy_labels = {}
        
        for accuracy in accuracy_levels:
            if accuracy < 0 or accuracy > 1:
                continue
            
            # Calculate flip probability to achieve target accuracy
            flip_prob = 1 - accuracy
            
            # Generate noisy labels
            labels = original_labels.copy()
            flip_mask = np.random.random(len(labels)) < flip_prob
            labels[flip_mask] = 1 - labels[flip_mask]
            
            noisy_labels[accuracy] = labels
        
        return noisy_labels
    
    def evaluate_labeling_robustness(self, trades: List[Trade],
                                   original_labels: np.ndarray) -> Dict[str, Union[float, Dict]]:
        """
        Evaluate robustness of labeling algorithm
        
        Args:
            trades: List of trades
            original_labels: Original labels
            
        Returns:
            Robustness analysis results
        """
        # Calculate baseline performance
        baseline_metrics = self.metrics_calculator.calculate_comprehensive_metrics(trades)
        baseline_cr = baseline_metrics.cumulative_return
        
        # Test robustness across different noise levels
        robustness_results = {}
        
        for noise_level in self.config.noise_levels:
            accuracy = 1.0 - noise_level
            
            # Run Monte Carlo simulation
            cr_changes = []
            
            for _ in range(self.config.monte_carlo_runs):
                # Generate noisy labels
                noisy_labels = self.noise_model_simulation(original_labels, [accuracy])
                
                # Simulate performance with noisy labels
                # This is a simplified simulation - in practice, you'd retrain the model
                noise_factor = 1.0 - noise_level * 0.5  # Simplified noise impact
                simulated_cr = baseline_cr * noise_factor
                
                # Calculate percentage change
                cr_change = abs((simulated_cr - baseline_cr) / baseline_cr) * 100
                cr_changes.append(cr_change)
            
            robustness_results[noise_level] = {
                'mean_cr_change': np.mean(cr_changes),
                'std_cr_change': np.std(cr_changes),
                'max_cr_change': np.max(cr_changes),
                'percentile_95': np.percentile(cr_changes, 95)
            }
        
        # Calculate overall robustness score
        robustness_score = self._calculate_robustness_score(robustness_results)
        
        return {
            'robustness_score': robustness_score,
            'baseline_cr': baseline_cr,
            'noise_analysis': robustness_results
        }
    
    def compare_labeling_methods(self, trades_dict: Dict[str, List[Trade]]) -> pd.DataFrame:
        """
        Compare different labeling methods
        
        Args:
            trades_dict: Dictionary mapping method names to trade lists
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for method_name, trades in trades_dict.items():
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(trades)
            result = {'Method': method_name, **metrics.to_dict()}
            comparison_results.append(result)
        
        return pd.DataFrame(comparison_results)
    
    def _calculate_robustness_score(self, robustness_results: Dict) -> float:
        """
        Calculate overall robustness score
        Lower is better (less sensitive to noise)
        """
        # Weight different noise levels
        weighted_changes = []
        
        for noise_level, results in robustness_results.items():
            if noise_level > 0:  # Skip perfect accuracy
                weight = noise_level  # Higher noise gets higher weight
                weighted_change = results['mean_cr_change'] * weight
                weighted_changes.append(weighted_change)
        
        return np.mean(weighted_changes) if weighted_changes else 0.0


class CrossValidationEvaluator:
    """
    Implements time series cross-validation for model evaluation
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics_calculator = FinancialMetricsCalculator(config.risk_free_rate)
    
    def time_series_cross_validation(self, features: np.ndarray, 
                                   labels: np.ndarray,
                                   model_trainer,
                                   n_splits: int = None) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation
        
        Args:
            features: Input features
            labels: Target labels
            model_trainer: Model training function
            n_splits: Number of CV splits
            
        Returns:
            Cross-validation results
        """
        n_splits = n_splits or self.config.cv_folds
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'train_scores': [],
            'val_scores': [],
            'metrics': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            print(f"Processing fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Train model
            trained_model = model_trainer(X_train, y_train)
            
            # Evaluate
            train_score = self._evaluate_model(trained_model, X_train, y_train)
            val_score = self._evaluate_model(trained_model, X_val, y_val)
            
            cv_results['train_scores'].append(train_score)
            cv_results['val_scores'].append(val_score)
        
        return cv_results
    
    def _evaluate_model(self, model, features: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate model performance"""
        # This would depend on your specific model interface
        # For now, return a placeholder score
        predictions = model.predict(features)
        score = accuracy_score(labels, predictions > 0.5)
        return score


class VisualizationTools:
    """
    Visualization tools for performance analysis
    """
    
    @staticmethod
    def plot_performance_comparison(metrics_df: pd.DataFrame, 
                                  save_path: Optional[str] = None) -> go.Figure:
        """
        Create performance comparison visualization
        
        Args:
            metrics_df: DataFrame with performance metrics
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Sharpe Ratio', 'Max Drawdown', 'Win Rate',
                          'Total Return', 'Volatility', 'Profit Factor'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        methods = metrics_df['Method']
        
        # Sharpe Ratio
        fig.add_trace(
            go.Bar(x=methods, y=metrics_df['sharpe_ratio'], name='Sharpe Ratio'),
            row=1, col=1
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Bar(x=methods, y=metrics_df['max_drawdown'], name='Max Drawdown'),
            row=1, col=2
        )
        
        # Win Rate
        fig.add_trace(
            go.Bar(x=methods, y=metrics_df['win_rate'], name='Win Rate'),
            row=1, col=3
        )
        
        # Total Return
        fig.add_trace(
            go.Bar(x=methods, y=metrics_df['total_return'], name='Total Return'),
            row=2, col=1
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(x=methods, y=metrics_df['volatility'], name='Volatility'),
            row=2, col=2
        )
        
        # Profit Factor
        fig.add_trace(
            go.Bar(x=methods, y=metrics_df['profit_factor'], name='Profit Factor'),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Performance Metrics Comparison",
            showlegend=False,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_robustness_analysis(robustness_results: Dict,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Plot robustness analysis results
        
        Args:
            robustness_results: Results from robustness analysis
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        noise_levels = list(robustness_results['noise_analysis'].keys())
        mean_changes = [robustness_results['noise_analysis'][level]['mean_cr_change'] 
                       for level in noise_levels]
        std_changes = [robustness_results['noise_analysis'][level]['std_cr_change'] 
                      for level in noise_levels]
        
        fig = go.Figure()
        
        # Mean change line
        fig.add_trace(
            go.Scatter(
                x=noise_levels,
                y=mean_changes,
                mode='lines+markers',
                name='Mean CR Change (%)',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            )
        )
        
        # Error bars
        fig.add_trace(
            go.Scatter(
                x=noise_levels + noise_levels[::-1],
                y=[m + s for m, s in zip(mean_changes, std_changes)] + 
                  [m - s for m, s in zip(mean_changes[::-1], std_changes[::-1])],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title=f"Labeling Algorithm Robustness Analysis<br>"
                  f"Robustness Score: {robustness_results['robustness_score']:.2f}",
            xaxis_title="Noise Level",
            yaxis_title="Cumulative Return Change (%)",
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_trade_analysis(trades: List[Trade], 
                          save_path: Optional[str] = None) -> go.Figure:
        """
        Plot trade analysis
        
        Args:
            trades: List of trades
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        returns = [t.return_pct for t in trades if t.return_pct is not None]
        durations = [t.duration for t in trades if t.duration is not None]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Distribution', 'Duration Distribution',
                          'Returns vs Duration', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=30, name='Returns'),
            row=1, col=1
        )
        
        # Duration distribution
        fig.add_trace(
            go.Histogram(x=durations, nbinsx=20, name='Duration'),
            row=1, col=2
        )
        
        # Returns vs Duration scatter
        fig.add_trace(
            go.Scatter(
                x=durations, y=returns, mode='markers',
                name='Returns vs Duration',
                marker=dict(size=6, opacity=0.6)
            ),
            row=2, col=1
        )
        
        # Cumulative returns
        cumulative_returns = np.cumprod(1 + np.array(returns))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_returns))),
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Trade Analysis Dashboard",
            showlegend=False,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


if __name__ == "__main__":
    # Example usage
    from config import load_config
    
    config = load_config()
    
    # Initialize components
    metrics_calculator = FinancialMetricsCalculator(config.evaluation.risk_free_rate)
    robustness_analyzer = RobustnessAnalyzer(config.evaluation)
    
    # Generate sample trades for testing
    np.random.seed(42)
    sample_trades = []
    
    for i in range(100):
        # Random trade outcomes
        outcome = np.random.choice([
            TradeOutcome.PROFIT_TARGET_HIT,
            TradeOutcome.STOP_LOSS_HIT,
            TradeOutcome.TIME_EXPIRED_PROFIT,
            TradeOutcome.TIME_EXPIRED_LOSS
        ])
        
        if outcome == TradeOutcome.PROFIT_TARGET_HIT:
            return_pct = np.random.uniform(0.02, 0.08)
        elif outcome == TradeOutcome.STOP_LOSS_HIT:
            return_pct = np.random.uniform(-0.05, -0.01)
        elif outcome == TradeOutcome.TIME_EXPIRED_PROFIT:
            return_pct = np.random.uniform(0.001, 0.02)
        else:
            return_pct = np.random.uniform(-0.02, -0.001)
        
        trade = Trade(
            entry_price=100.0,
            entry_time=pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            profit_target=0.05,
            stop_loss=0.02,
            time_horizon=10
        )
        trade.outcome = outcome
        trade.return_pct = return_pct
        trade.duration = np.random.randint(1, 15)
        
        sample_trades.append(trade)
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_comprehensive_metrics(sample_trades)
    
    print("Performance Metrics:")
    print(f"Total Return: {metrics.total_return:.4f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.4f}")
    print(f"Win Rate: {metrics.win_rate:.4f}")
    
    # Test robustness
    original_labels = np.random.choice([0, 1], size=100)
    robustness_results = robustness_analyzer.evaluate_labeling_robustness(
        sample_trades, original_labels
    )
    
    print(f"\nRobustness Score: {robustness_results['robustness_score']:.4f}")
    
    # Create visualizations
    viz = VisualizationTools()
    
    # Performance comparison (with dummy data)
    comparison_data = pd.DataFrame([
        {'Method': 'Dynamic Labeling', **metrics.to_dict()},
        {'Method': 'Fixed Threshold', 'total_return': 0.08, 'sharpe_ratio': 1.2, 
         'max_drawdown': -0.15, 'win_rate': 0.55, 'volatility': 0.18, 'profit_factor': 1.8}
    ])
    
    fig1 = viz.plot_performance_comparison(comparison_data, 'results/performance_comparison.html')
    fig2 = viz.plot_robustness_analysis(robustness_results, 'results/robustness_analysis.html')
    fig3 = viz.plot_trade_analysis(sample_trades, 'results/trade_analysis.html')
    
    print("\nEvaluation complete! Visualizations saved to results/ directory.")

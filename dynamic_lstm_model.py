"""
Dynamic Labeling LSTM Model for Stock Prediction
Implements an LSTM-CNN encoder-decoder architecture with attention mechanism
to predict profit-taking thresholds, stop-loss thresholds, and time horizons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Compute scaled dot-product attention
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights


class CNNFeatureExtractor(nn.Module):
    """
    CNN component for spatial feature extraction
    """
    
    def __init__(self, input_dim: int, cnn_filters: List[int], 
                 kernel_sizes: List[int], dropout: float = 0.2):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        
        in_channels = 1
        for filters, kernel_size in zip(cnn_filters, kernel_sizes):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2, stride=1, padding=1)
            )
            self.conv_layers.append(conv_block)
            in_channels = filters
        
        # Calculate output dimension
        self.output_dim = cnn_filters[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        # Reshape for conv1d: (batch_size, 1, seq_len * input_dim)
        batch_size, seq_len, input_dim = x.shape
        x = x.view(batch_size, 1, seq_len * input_dim)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)  # (batch_size, cnn_filters[-1])
        
        return x


class DynamicLabelingLSTM(nn.Module):
    """
    Main LSTM model for dynamic labeling prediction
    Predicts profit-taking threshold, stop-loss threshold, and time horizon
    """
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.lstm_hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.lstm_hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            input_dim=input_dim,
            cnn_filters=config.cnn_filters,
            kernel_sizes=config.cnn_kernel_sizes,
            dropout=config.lstm_dropout
        )
        
        # Attention mechanism
        if config.use_attention:
            self.attention = MultiHeadAttention(
                d_model=config.lstm_hidden_size * 2,  # Bidirectional LSTM
                n_heads=8,
                dropout=config.lstm_dropout
            )
        
        # Feature fusion
        lstm_output_dim = config.lstm_hidden_size * 2  # Bidirectional
        cnn_output_dim = self.cnn_extractor.output_dim
        fusion_dim = lstm_output_dim + cnn_output_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout)
        )
        
        # Output heads for three predictions
        hidden_dim = config.lstm_hidden_size // 2
        
        # Profit-taking threshold (0.01 to 0.15, i.e., 1% to 15%)
        self.pt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Scale to [0, 1] then transform to [pt_min, pt_max]
        )
        
        # Stop-loss threshold (0.005 to 0.10, i.e., 0.5% to 10%)
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Scale to [0, 1] then transform to [sl_min, sl_max]
        )
        
        # Time horizon (1 to 30 periods)
        self.th_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Scale to [0, 1] then transform to [th_min, th_max]
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.lstm_dropout)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions for PT, SL, and TH
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Input projection and positional encoding
        x_proj = self.input_projection(x)
        x_proj = self.pos_encoding(x_proj.transpose(0, 1)).transpose(0, 1)
        
        # LSTM encoding
        lstm_output, (hidden, cell) = self.lstm(x_proj)
        
        # Apply attention if configured
        if self.config.use_attention:
            attended_output, attention_weights = self.attention(
                lstm_output, lstm_output, lstm_output
            )
            # Use the last time step
            lstm_features = attended_output[:, -1, :]
        else:
            # Use the last time step
            lstm_features = lstm_output[:, -1, :]
        
        # CNN feature extraction
        cnn_features = self.cnn_extractor(x)
        
        # Feature fusion
        combined_features = torch.cat([lstm_features, cnn_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.dropout(fused_features)
        
        # Predictions
        pt_raw = self.pt_head(fused_features)
        sl_raw = self.sl_head(fused_features)
        th_raw = self.th_head(fused_features)
        
        return {
            'pt_raw': pt_raw,
            'sl_raw': sl_raw,
            'th_raw': th_raw,
            'lstm_features': lstm_features,
            'cnn_features': cnn_features,
            'fused_features': fused_features
        }
    
    def predict(self, x: torch.Tensor, pt_range: Tuple[float, float] = (0.01, 0.15),
                sl_range: Tuple[float, float] = (0.005, 0.10),
                th_range: Tuple[int, int] = (1, 30)) -> Dict[str, torch.Tensor]:
        """
        Make predictions and scale to appropriate ranges
        
        Args:
            x: Input tensor
            pt_range: (min, max) for profit-taking threshold
            sl_range: (min, max) for stop-loss threshold
            th_range: (min, max) for time horizon
            
        Returns:
            Scaled predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Scale predictions to appropriate ranges
            pt_min, pt_max = pt_range
            sl_min, sl_max = sl_range
            th_min, th_max = th_range
            
            pt_pred = pt_min + outputs['pt_raw'] * (pt_max - pt_min)
            sl_pred = sl_min + outputs['sl_raw'] * (sl_max - sl_min)
            th_pred = th_min + outputs['th_raw'] * (th_max - th_min)
            
            return {
                'profit_taking': pt_pred,
                'stop_loss': sl_pred,
                'time_horizon': th_pred.round().int(),  # Round to integer days
                'raw_outputs': outputs
            }


class DynamicLabelingLoss(nn.Module):
    """
    Custom loss function for dynamic labeling
    Combines multiple objectives with financial relevance
    """
    
    def __init__(self, pt_weight: float = 1.0, sl_weight: float = 1.0, 
                 th_weight: float = 1.0, consistency_weight: float = 0.1):
        super().__init__()
        
        self.pt_weight = pt_weight
        self.sl_weight = sl_weight
        self.th_weight = th_weight
        self.consistency_weight = consistency_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Loss components and total loss
        """
        # Individual losses
        pt_loss = self.mse_loss(predictions['pt_raw'], targets['pt'])
        sl_loss = self.mse_loss(predictions['sl_raw'], targets['sl'])
        th_loss = self.mse_loss(predictions['th_raw'], targets['th'])
        
        # Consistency constraint: PT should generally be > SL
        consistency_loss = F.relu(predictions['sl_raw'] - predictions['pt_raw']).mean()
        
        # Combined loss
        total_loss = (
            self.pt_weight * pt_loss +
            self.sl_weight * sl_loss +
            self.th_weight * th_loss +
            self.consistency_weight * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'pt_loss': pt_loss,
            'sl_loss': sl_loss,
            'th_loss': th_loss,
            'consistency_loss': consistency_loss
        }


class ModelTrainer:
    """
    Training class for the dynamic labeling LSTM model
    """
    
    def __init__(self, model: DynamicLabelingLSTM, config: ModelConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = DynamicLabelingLoss(
            pt_weight=config.pt_weight,
            sl_weight=config.sl_weight,
            th_weight=config.th_weight
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.patience // 2,
            factor=0.5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'pt_loss': 0.0,
            'sl_loss': 0.0,
            'th_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        num_batches = 0
        
        for batch_data in train_loader:
            features = batch_data['features'].to(self.device)
            targets = {
                'pt': batch_data['pt'].to(self.device),
                'sl': batch_data['sl'].to(self.device),
                'th': batch_data['th'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Compute loss
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Validate for one epoch
        """
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'pt_loss': 0.0,
            'sl_loss': 0.0,
            'th_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                features = batch_data['features'].to(self.device)
                targets = {
                    'pt': batch_data['pt'].to(self.device),
                    'sl': batch_data['sl'].to(self.device),
                    'th': batch_data['th'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(features)
                
                # Compute loss
                losses = self.criterion(predictions, targets)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, train_loader, val_loader, num_epochs: int = None) -> Dict[str, List[float]]:
        """
        Full training loop
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_losses['total_loss'])
            
            # Save losses
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            # Early stopping check
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            else:
                self.patience_counter += 1
                
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_losses['total_loss']:.6f}, "
                      f"Val Loss: {val_losses['total_loss']:.6f}")
                print(f"PT: {train_losses['pt_loss']:.6f} -> {val_losses['pt_loss']:.6f}, "
                      f"SL: {train_losses['sl_loss']:.6f} -> {val_losses['sl_loss']:.6f}, "
                      f"TH: {train_losses['th_loss']:.6f} -> {val_losses['th_loss']:.6f}")
                
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_model.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


if __name__ == "__main__":
    # Example usage
    from config import load_config
    
    config = load_config()
    model_config = config.model
    
    # Initialize model
    input_dim = 50  # Example feature dimension
    model = DynamicLabelingLSTM(model_config, input_dim)
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    seq_len = 60
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        outputs = model(x)
        predictions = model.predict(x)
    
    print("Forward pass successful!")
    print(f"Predictions shape - PT: {predictions['profit_taking'].shape}, "
          f"SL: {predictions['stop_loss'].shape}, TH: {predictions['time_horizon'].shape}")

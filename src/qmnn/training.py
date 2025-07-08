"""
Training utilities for QMNN models.

This module provides training loops, optimization strategies, and
evaluation metrics for quantum memory-augmented neural networks.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import logging

from .models import QMNN


class QMNNTrainer:
    """
    Trainer class for Quantum Memory-Augmented Neural Networks.
    """
    
    def __init__(
        self,
        model: QMNN,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        log_mlflow: bool = True,
    ):
        """
        Initialize QMNN trainer.
        
        Args:
            model: QMNN model to train
            device: Device to use ('cpu', 'cuda', or 'auto')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            gradient_clip: Gradient clipping threshold
            log_mlflow: Whether to log to MLflow
        """
        self.model = model
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Optimizer setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Training parameters
        self.gradient_clip = gradient_clip
        self.log_mlflow = log_mlflow
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        if self.log_mlflow:
            mlflow.set_experiment("QMNN_Training")
            
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Add sequence dimension if needed
            if len(data.shape) == 2:
                data = data.unsqueeze(1)  # [batch, 1, features]
                
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, memory_content = self.model(data, store_memory=True)
            
            # Remove sequence dimension for loss computation
            if len(outputs.shape) == 3:
                outputs = outputs.squeeze(1)
                
            loss = criterion(outputs, targets)
            
            # Add memory regularization
            memory_usage = self.model.memory_usage()
            memory_reg = 0.01 * memory_usage  # Encourage memory usage
            loss += memory_reg
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
                
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            if len(targets.shape) == 1:  # Classification
                predicted = outputs.argmax(dim=1)
                total_correct += (predicted == targets).sum().item()
            else:  # Regression
                total_correct += ((outputs - targets).abs() < 0.1).float().mean().item()
                
            total_samples += targets.size(0)
            
            # Update progress bar
            current_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'Mem': f'{memory_usage:.2f}'
            })
            
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_samples
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'memory_usage': self.model.memory_usage(),
        }
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        return metrics
        
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Validate model performance.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Add sequence dimension if needed
                if len(data.shape) == 2:
                    data = data.unsqueeze(1)
                    
                outputs, _ = self.model(data, store_memory=False)
                
                # Remove sequence dimension for loss computation
                if len(outputs.shape) == 3:
                    outputs = outputs.squeeze(1)
                    
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                if len(targets.shape) == 1:  # Classification
                    predicted = outputs.argmax(dim=1)
                    total_correct += (predicted == targets).sum().item()
                else:  # Regression
                    total_correct += ((outputs - targets).abs() < 0.1).float().mean().item()
                    
                total_samples += targets.size(0)
                
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_correct / total_samples
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy,
        }
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        checkpoint_path: str = "qmnn_checkpoint.pth",
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            criterion: Loss function
            scheduler: Learning rate scheduler
            early_stopping_patience: Early stopping patience
            save_best: Whether to save best model
            checkpoint_path: Path to save checkpoints
            
        Returns:
            Dictionary of training history
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        best_val_loss = float('inf')
        patience_counter = 0
        
        if self.log_mlflow:
            mlflow.start_run()
            mlflow.log_params({
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'num_epochs': num_epochs,
                'model_type': 'QMNN',
                'memory_capacity': self.model.memory_capacity,
                'memory_embedding_dim': self.model.memory_embedding_dim,
            })
            
        try:
            for epoch in range(num_epochs):
                # Training
                train_metrics = self.train_epoch(train_loader, criterion, epoch)
                
                # Validation
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self.validate(val_loader, criterion)
                    
                # Learning rate scheduling
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                    else:
                        scheduler.step()
                        
                # Logging
                all_metrics = {**train_metrics, **val_metrics}
                
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f}"
                )
                
                if val_loader is not None:
                    self.logger.info(
                        f"Val Loss: {val_metrics['val_loss']:.4f}, "
                        f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                    )
                    
                if self.log_mlflow:
                    mlflow.log_metrics(all_metrics, step=epoch)
                    
                # Early stopping and checkpointing
                current_val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    
                    if save_best:
                        self.save_checkpoint(checkpoint_path, epoch, all_metrics)
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
        finally:
            if self.log_mlflow:
                mlflow.end_run()
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'memory_capacity': self.model.memory_capacity,
                'memory_embedding_dim': self.model.memory_embedding_dim,
            }
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def main():
    """CLI entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train QMNN model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--memory-capacity', type=int, default=256, help='Memory capacity')
    
    args = parser.parse_args()
    
    # Implementation would load data and start training
    print(f"Training QMNN with {args.epochs} epochs...")


if __name__ == "__main__":
    main()

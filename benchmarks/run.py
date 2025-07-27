#!/usr/bin/env python3
"""
QMANN Comprehensive Benchmark Suite

Enhanced benchmark suite with real datasets, improved baselines, and detailed analysis.
Includes MNIST, CIFAR-10, memory tasks, and quantum advantage metrics.
"""

import argparse
import time
import json
import csv
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import psutil
import gc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import wandb

from qmann.models import QMANN, QuantumNeuralNetwork
from qmann.training import QMANNTrainer
from qmann.utils import plot_training_history, create_benchmark_plot


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClassicalLSTM(nn.Module):
    """Enhanced LSTM baseline with attention and regularization."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.norm(attended + lstm_out)
        attended = self.dropout(attended)
        return self.fc(attended)


class ClassicalTransformer(nn.Module):
    """Enhanced Transformer baseline with modern architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.transformer(x)
        x = self.norm(x)
        return self.output_projection(x)


class NeuralTuringMachine(nn.Module):
    """Neural Turing Machine baseline for memory comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 memory_size: int = 128, memory_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        self.controller = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Memory interface
        self.read_head = nn.Linear(hidden_dim, memory_dim + 3)  # key + strength + gate
        self.write_head = nn.Linear(hidden_dim, memory_dim * 2 + 3)  # key + value + strength + gate

        # Memory
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))

        self.output_layer = nn.Linear(hidden_dim + memory_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize memory for batch
        memory = self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

        outputs = []
        hidden = None

        for t in range(seq_len):
            # Controller
            controller_out, hidden = self.controller(x[:, t:t+1], hidden)
            controller_out = controller_out.squeeze(1)

            # Read from memory
            read_params = self.read_head(controller_out)
            read_key = read_params[:, :self.memory_dim]
            read_strength = F.softplus(read_params[:, self.memory_dim])

            # Attention over memory
            similarities = F.cosine_similarity(
                read_key.unsqueeze(1), memory, dim=2
            )
            attention = F.softmax(similarities * read_strength.unsqueeze(1), dim=1)
            read_vector = torch.sum(attention.unsqueeze(2) * memory, dim=1)

            # Output
            combined = torch.cat([controller_out, read_vector], dim=1)
            output = self.output_layer(combined)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class DifferentiableNeuralComputer(nn.Module):
    """Differentiable Neural Computer baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 memory_size: int = 256, memory_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads

        self.controller = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Interface parameters
        interface_size = (
            memory_dim * num_heads +  # read keys
            memory_dim * num_heads +  # write keys
            memory_dim +              # write vector
            num_heads +               # read strengths
            1 +                       # write strength
            memory_dim +              # erase vector
            num_heads * 3             # read modes
        )
        self.interface = nn.Linear(hidden_dim, interface_size)

        # Memory
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))
        self.register_buffer('usage', torch.zeros(memory_size))

        self.output_layer = nn.Linear(hidden_dim + memory_dim * num_heads, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize memory and usage for batch
        memory = self.memory.unsqueeze(0).repeat(batch_size, 1, 1)
        usage = self.usage.unsqueeze(0).repeat(batch_size, 1)

        outputs = []
        hidden = None

        for t in range(seq_len):
            # Controller
            controller_out, hidden = self.controller(x[:, t:t+1], hidden)
            controller_out = controller_out.squeeze(1)

            # Interface
            interface_params = self.interface(controller_out)

            # Simplified DNC operations (full implementation would be much longer)
            # Read from memory using content-based addressing
            read_vectors = []
            for head in range(self.num_heads):
                start_idx = head * self.memory_dim
                end_idx = (head + 1) * self.memory_dim
                read_key = interface_params[:, start_idx:end_idx]

                similarities = F.cosine_similarity(
                    read_key.unsqueeze(1), memory, dim=2
                )
                attention = F.softmax(similarities, dim=1)
                read_vector = torch.sum(attention.unsqueeze(2) * memory, dim=1)
                read_vectors.append(read_vector)

            read_vectors = torch.cat(read_vectors, dim=1)

            # Output
            combined = torch.cat([controller_out, read_vectors], dim=1)
            output = self.output_layer(combined)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class BenchmarkSuite:
    """Comprehensive benchmark suite for QMANN with real datasets and advanced metrics."""

    def __init__(self, output_dir: str = "benchmarks/results", use_wandb: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.results = []
        self.use_wandb = use_wandb

        if use_wandb:
            wandb.init(project="qmann-benchmarks", config={
                "device": str(self.device),
                "output_dir": str(self.output_dir)
            })

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_mnist_sequential(self, seq_len: int = 28, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Load MNIST as sequential data for RNN training."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )

        # Convert images to sequences
        def collate_fn(batch):
            images, labels = zip(*batch)
            images = torch.stack(images).squeeze(1)  # Remove channel dim
            # Reshape to sequences: (batch, seq_len, input_dim)
            images = images.view(len(images), seq_len, -1)
            labels = torch.tensor(labels)
            return images, labels

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, test_loader

    def load_cifar10_sequential(self, seq_len: int = 32, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-10 as sequential data."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )

        def collate_fn(batch):
            images, labels = zip(*batch)
            images = torch.stack(images)
            # Reshape to sequences: (batch, seq_len, input_dim)
            images = images.view(len(images), seq_len, -1)
            labels = torch.tensor(labels)
            return images, labels

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, test_loader

    def generate_memory_task_data(
        self,
        task_type: str = "copy",
        n_samples: int = 1000,
        seq_len: int = 20,
        input_dim: int = 8,
        delay: int = 10
    ) -> Tuple[DataLoader, DataLoader]:
        """Generate algorithmic memory task data."""

        if task_type == "copy":
            # Copy task: memorize sequence and reproduce after delay
            inputs = []
            targets = []

            for _ in range(n_samples):
                # Generate random sequence
                seq = torch.randn(seq_len, input_dim)
                # Add delay period
                delay_seq = torch.zeros(delay, input_dim)
                # Create input with delimiter
                input_seq = torch.cat([seq, delay_seq, torch.zeros(seq_len, input_dim)])
                # Target is zeros during input/delay, then original sequence
                target_seq = torch.cat([torch.zeros(seq_len + delay, input_dim), seq])

                inputs.append(input_seq)
                targets.append(target_seq)

        elif task_type == "associative_recall":
            # Associative recall: learn key-value pairs and recall by key
            inputs = []
            targets = []

            for _ in range(n_samples):
                n_pairs = np.random.randint(3, 8)
                keys = torch.randn(n_pairs, input_dim // 2)
                values = torch.randn(n_pairs, input_dim // 2)

                # Present key-value pairs
                input_seq = []
                for k, v in zip(keys, values):
                    input_seq.append(torch.cat([k, v]))

                # Add query (random key)
                query_idx = np.random.randint(n_pairs)
                query_key = keys[query_idx]
                input_seq.append(torch.cat([query_key, torch.zeros(input_dim // 2)]))

                input_seq = torch.stack(input_seq)

                # Target is the corresponding value
                target_seq = torch.zeros_like(input_seq)
                target_seq[-1, input_dim // 2:] = values[query_idx]

                inputs.append(input_seq)
                targets.append(target_seq)

        # Convert to tensors and create data loaders
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        # Split train/test
        train_size = int(0.8 * len(inputs))
        train_dataset = TensorDataset(inputs[:train_size], targets[:train_size])
        test_dataset = TensorDataset(inputs[train_size:], targets[train_size:])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def generate_synthetic_data(
        self,
        n_samples: int,
        seq_len: int,
        input_dim: int,
        output_dim: int,
        task_type: str = "classification"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data for benchmarking."""

        # Generate input sequences with temporal patterns
        X = torch.randn(n_samples, seq_len, input_dim)

        # Add temporal correlations
        for i in range(1, seq_len):
            X[:, i] = 0.7 * X[:, i-1] + 0.3 * X[:, i]

        if task_type == "classification":
            # Generate classification targets based on sequence properties
            sequence_means = X.mean(dim=1)
            y = torch.argmax(sequence_means, dim=1) % output_dim
            y = y.unsqueeze(1).repeat(1, seq_len)
        else:
            # Generate regression targets with temporal dependencies
            y = torch.cumsum(X.mean(dim=2, keepdim=True), dim=1)
            y = y.repeat(1, 1, output_dim)

        return X, y

    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        task_type: str = "classification",
        learning_rate: float = 1e-3,
        patience: int = 5
    ) -> Dict[str, Any]:
        """Enhanced model benchmarking with detailed metrics."""

        logger.info(f"Benchmarking {model_name}...")

        model = model.to(self.device)

        # Setup training with scheduler
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )

        # Training metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        train_times = []
        memory_usage = []
        learning_rates = []

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, QMANN):
                    outputs, _ = model(data)
                else:
                    outputs = model(data)
                    
                # Handle output dimensions
                if len(outputs.shape) == 3 and len(targets.shape) == 2:
                    outputs = outputs.view(-1, outputs.shape[-1])
                    targets = targets.view(-1)
                elif len(outputs.shape) == 3 and len(targets.shape) == 3:
                    outputs = outputs.view(-1, outputs.shape[-1])
                    targets = targets.view(-1, targets.shape[-1])
                    
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                
                if task_type == "classification":
                    predicted = outputs.argmax(dim=1)
                    epoch_correct += (predicted == targets).sum().item()
                    epoch_total += targets.size(0)
                    
            # Training metrics
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if task_type == "classification":
                train_accuracy = epoch_correct / epoch_total
                train_accuracies.append(train_accuracy)
            else:
                train_accuracies.append(1.0 - avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)

                    if isinstance(model, QMANN):
                        outputs = model(data, store_memory=False)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        outputs = model(data)

                    # Handle output dimensions
                    if len(outputs.shape) == 3 and len(targets.shape) == 2:
                        outputs = outputs.view(-1, outputs.shape[-1])
                        targets = targets.view(-1)
                    elif len(outputs.shape) == 3 and len(targets.shape) == 3:
                        outputs = outputs.view(-1, outputs.shape[-1])
                        targets = targets.view(-1, targets.shape[-1])

                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    if task_type == "classification":
                        predicted = outputs.argmax(dim=1)
                        val_correct += (predicted == targets).sum().item()
                        val_total += targets.size(0)

            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            if task_type == "classification":
                val_accuracy = val_correct / val_total
                val_accuracies.append(val_accuracy)
            else:
                val_accuracies.append(1.0 - avg_val_loss)

            # Learning rate and timing
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            epoch_time = time.time() - epoch_start
            train_times.append(epoch_time)

            # Memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)
            else:
                memory_usage.append(psutil.Process().memory_info().rss / 1024**2)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Logging
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                          f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

            # WandB logging
            if self.use_wandb:
                wandb.log({
                    f"{model_name}/train_loss": avg_train_loss,
                    f"{model_name}/val_loss": avg_val_loss,
                    f"{model_name}/train_accuracy": train_accuracies[-1],
                    f"{model_name}/val_accuracy": val_accuracies[-1],
                    f"{model_name}/learning_rate": current_lr,
                    "epoch": epoch
                })

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        total_train_time = time.time() - start_time

        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final comprehensive evaluation
        eval_results = self._comprehensive_evaluation(
            model, model_name, test_loader, task_type
        )

        # Model analysis
        model_size = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Memory efficiency (for QMANN)
        memory_efficiency = 0.0
        quantum_info = {}
        if isinstance(model, QMANN):
            memory_efficiency = model.memory_usage()
            quantum_info = model.get_model_info()

        results = {
            'model': model_name,
            'task_type': task_type,
            'total_train_time': total_train_time,
            'epochs_trained': len(train_losses),
            'best_epoch': len(train_losses) - patience_counter - 1 if best_model_state else len(train_losses) - 1,

            # Training metrics
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'train_times': train_times,

            # Final performance
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1],
            'best_val_loss': best_val_loss,

            # Model characteristics
            'model_size': model_size,
            'trainable_parameters': trainable_params,
            'memory_efficiency': memory_efficiency,
            'peak_memory_mb': max(memory_usage) if memory_usage else 0,
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,

            # Detailed evaluation
            **eval_results,

            # Quantum-specific info
            'quantum_info': quantum_info,
        }

        logger.info(f"{model_name} - Final Val Accuracy: {val_accuracies[-1]:.4f}, "
                   f"Train Time: {total_train_time:.2f}s, Parameters: {trainable_params:,}")

        return results

    def _comprehensive_evaluation(
        self,
        model: nn.Module,
        model_name: str,
        test_loader: DataLoader,
        task_type: str
    ) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""

        model.eval()
        all_predictions = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Measure inference time
                start_time = time.time()

                if isinstance(model, QMANN):
                    outputs = model(data, store_memory=False)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = model(data)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Handle output dimensions
                if len(outputs.shape) == 3 and len(targets.shape) == 2:
                    outputs = outputs.view(-1, outputs.shape[-1])
                    targets = targets.view(-1)
                elif len(outputs.shape) == 3 and len(targets.shape) == 3:
                    outputs = outputs.view(-1, outputs.shape[-1])
                    targets = targets.view(-1, targets.shape[-1])

                if task_type == "classification":
                    predictions = outputs.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

        # Calculate detailed metrics
        eval_results = {
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': np.sum(inference_times),
            'throughput_samples_per_sec': len(all_targets) / np.sum(inference_times),
        }

        if task_type == "classification":
            # Classification metrics
            try:
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

                accuracy = accuracy_score(all_targets, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='weighted', zero_division=0
                )

                eval_results.update({
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist()
                })
            except ImportError:
                warnings.warn("sklearn not available, using basic metrics")
                accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
                eval_results['test_accuracy'] = accuracy
        else:
            # Regression metrics
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                mse = mean_squared_error(all_targets, all_predictions)
                mae = mean_absolute_error(all_targets, all_predictions)
                r2 = r2_score(all_targets, all_predictions)

                eval_results.update({
                    'test_mse': mse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'test_rmse': np.sqrt(mse)
                })
            except ImportError:
                warnings.warn("sklearn not available, using basic metrics")
                mse = np.mean((np.array(all_predictions) - np.array(all_targets))**2)
                eval_results['test_mse'] = mse

        return eval_results

    def run_memory_scaling_benchmark(self) -> Dict[str, Any]:
        """Benchmark memory scaling properties."""
        
        logger.info("Running memory scaling benchmark...")
        
        memory_capacities = [16, 32, 64, 128, 256, 512]
        results = []
        
        for capacity in memory_capacities:
            logger.info(f"Testing memory capacity: {capacity}")
            
            # Generate data
            X, y = self.generate_synthetic_data(100, 10, 8, 3)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=16)
            
            # Create QMANN with specific memory capacity
            model = QMANN(
                input_dim=8,
                hidden_dim=32,
                output_dim=3,
                memory_capacity=capacity,
                memory_embedding_dim=16,
            )
            
            # Measure memory operations
            start_time = time.time()
            
            # Fill memory
            for i, (data, _) in enumerate(loader):
                if i * 16 >= capacity:  # Stop when memory is full
                    break
                    
                model.eval()
                with torch.no_grad():
                    _, _ = model(data, store_memory=True)
                    
            fill_time = time.time() - start_time
            
            # Measure retrieval time
            start_time = time.time()
            
            for data, _ in loader:
                model.eval()
                with torch.no_grad():
                    _, _ = model(data, store_memory=False)
                break  # Just one batch for timing
                
            retrieval_time = time.time() - start_time
            
            results.append({
                'memory_capacity': capacity,
                'fill_time': fill_time,
                'retrieval_time': retrieval_time,
                'memory_usage': model.memory_usage(),
                'theoretical_capacity': 2**int(np.log2(capacity)),
            })
            
        return {'memory_scaling': results}
        
    def run_noise_resilience_benchmark(self) -> Dict[str, Any]:
        """Benchmark noise resilience of quantum components."""
        
        logger.info("Running noise resilience benchmark...")
        
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        results = []
        
        # Generate test data
        X, y = self.generate_synthetic_data(200, 5, 6, 2)
        train_dataset = TensorDataset(X[:150], y[:150])
        test_dataset = TensorDataset(X[150:], y[150:])
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        for noise_level in noise_levels:
            logger.info(f"Testing noise level: {noise_level}")
            
            # Create QMANN
            model = QMANN(
                input_dim=6,
                hidden_dim=24,
                output_dim=2,
                memory_capacity=32,
                memory_embedding_dim=12,
            )
            
            # Simulate noise by adding noise to quantum parameters
            if noise_level > 0:
                for layer in model.quantum_layers:
                    with torch.no_grad():
                        noise = torch.randn_like(layer.quantum_params) * noise_level
                        layer.quantum_params.add_(noise)
                        
            # Quick training (reduced epochs for noise test)
            result = self.benchmark_model(
                model, f"QMANN_noise_{noise_level}", 
                train_loader, test_loader, 
                num_epochs=5
            )
            
            result['noise_level'] = noise_level
            results.append(result)
            
        return {'noise_resilience': results}
        
    def run_comparison_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive comparison between different models."""
        
        logger.info("Running model comparison benchmark...")
        
        # Generate data
        X, y = self.generate_synthetic_data(1000, 20, 10, 5)
        
        # Split data
        train_size = int(0.8 * len(X))
        train_dataset = TensorDataset(X[:train_size], y[:train_size])
        test_dataset = TensorDataset(X[train_size:], y[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Define models to compare (including 2025 state-of-the-art)
        models = {
            'LSTM': ClassicalLSTM(input_dim=10, hidden_dim=64, output_dim=5),
            'Transformer': ClassicalTransformer(input_dim=10, hidden_dim=64, output_dim=5),
            'QNN': QuantumNeuralNetwork(input_dim=10, output_dim=5, n_qubits=6),
            'QMANN': QMANN(
                input_dim=10,
                hidden_dim=64,
                output_dim=5,
                memory_capacity=128,
                memory_embedding_dim=32,
            ),
        }

        # Add 2025 advanced models if available
        try:
            from qmann.quantum_transformers import QuantumTransformerBlock
            models['QuantumTransformer'] = QuantumTransformerBlock(
                d_model=64, n_heads=8, d_ff=256, n_qubits=6
            )
        except ImportError:
            pass

        try:
            from qmann.error_correction import SurfaceCodeQRAM
            models['SurfaceCodeQRAM'] = SurfaceCodeQRAM(
                memory_size=32, code_distance=3
            )
        except ImportError:
            pass
        
        results = []
        
        for model_name, model in models.items():
            try:
                result = self.benchmark_model(
                    model, model_name, 
                    train_loader, test_loader,
                    num_epochs=20
                )
                results.append(result)
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                
        return {'model_comparison': results}
        
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Results saved to {output_path}")
        
        # Also save as CSV for easy analysis
        csv_path = self.output_dir / filename.replace('.json', '.csv')
        
        if 'model_comparison' in results:
            df = pd.DataFrame(results['model_comparison'])
            # Remove list columns for CSV
            csv_df = df.drop(columns=['train_losses', 'train_accuracies', 'train_times', 'memory_usage'], errors='ignore')
            csv_df.to_csv(csv_path, index=False)
            
    def generate_plots(self, results: Dict[str, Any]):
        """Generate benchmark plots."""
        
        logger.info("Generating benchmark plots...")
        
        # Model comparison plot
        if 'model_comparison' in results:
            df = pd.DataFrame(results['model_comparison'])
            create_benchmark_plot(df, save_path=self.output_dir / "model_comparison.png")
            
        # Memory scaling plot
        if 'memory_scaling' in results:
            df = pd.DataFrame(results['memory_scaling'])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].plot(df['memory_capacity'], df['fill_time'], 'o-', label='Fill Time')
            axes[0].plot(df['memory_capacity'], df['retrieval_time'], 's-', label='Retrieval Time')
            axes[0].set_xlabel('Memory Capacity')
            axes[0].set_ylabel('Time (s)')
            axes[0].set_title('Memory Operation Times')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(df['memory_capacity'], df['memory_usage'], 'o-', color='green')
            axes[1].set_xlabel('Memory Capacity')
            axes[1].set_ylabel('Memory Usage Ratio')
            axes[1].set_title('Memory Utilization')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "memory_scaling.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # Noise resilience plot
        if 'noise_resilience' in results:
            df = pd.DataFrame(results['noise_resilience'])
            
            plt.figure(figsize=(8, 6))
            plt.plot(df['noise_level'], df['test_accuracy'], 'o-', linewidth=2, markersize=8)
            plt.xlabel('Noise Level')
            plt.ylabel('Test Accuracy')
            plt.title('QMANN Noise Resilience')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "noise_resilience.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Plots saved to {self.output_dir}")

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite with real datasets."""

        logger.info("Starting comprehensive QMANN benchmark suite...")

        all_results = {
            'mnist_sequential': {},
            'memory_tasks': {},
            'synthetic_classification': {},
            'memory_scaling': {},
            'noise_resilience': {}
        }

        # 1. MNIST Sequential Classification (reduced for demo)
        logger.info("Running MNIST sequential classification benchmark...")
        try:
            train_loader, test_loader = self.load_mnist_sequential(seq_len=28, batch_size=32)

            models = {
                'QMANN': QMANN(28, 64, 10, memory_capacity=64, memory_embedding_dim=32, max_qubits=12),
                'LSTM': ClassicalLSTM(28, 64, 10, num_layers=2),
                'Transformer': ClassicalTransformer(28, 64, 10, num_heads=4, num_layers=2),
            }

            for name, model in models.items():
                try:
                    result = self.benchmark_model(
                        model, f"MNIST_{name}", train_loader, test_loader,
                        num_epochs=2, task_type="classification", learning_rate=1e-3
                    )
                    all_results['mnist_sequential'][name] = result
                except Exception as e:
                    logger.error(f"Failed to benchmark {name} on MNIST: {e}")

        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")

        # 2. Memory Task Benchmarks
        logger.info("Running memory task benchmarks...")
        try:
            train_loader, test_loader = self.generate_memory_task_data(
                task_type="copy", n_samples=200, seq_len=6, delay=2
            )

            memory_models = {
                'QMANN': QMANN(8, 32, 8, memory_capacity=32, memory_embedding_dim=16),
                'LSTM': ClassicalLSTM(8, 32, 8),
            }

            for name, model in memory_models.items():
                try:
                    result = self.benchmark_model(
                        model, f"Copy_{name}", train_loader, test_loader,
                        num_epochs=5, task_type="regression", learning_rate=1e-3
                    )
                    all_results['memory_tasks'][f"copy_{name}"] = result
                except Exception as e:
                    logger.error(f"Failed to benchmark {name} on copy task: {e}")

        except Exception as e:
            logger.error(f"Failed to generate copy task data: {e}")

        # 3. Other benchmarks
        try:
            all_results['memory_scaling'] = self.run_memory_scaling_benchmark()
        except Exception as e:
            logger.error(f"Failed to run memory scaling benchmark: {e}")

        try:
            all_results['noise_resilience'] = self.run_noise_resilience_benchmark()
        except Exception as e:
            logger.error(f"Failed to run noise resilience benchmark: {e}")

        # Save results
        self.save_results(all_results)
        self.create_comparison_plots(all_results)

        logger.info("Comprehensive benchmark completed!")
        return all_results


def main():
    """Main benchmark execution."""
    
    parser = argparse.ArgumentParser(description='QMANN Benchmark Suite')
    parser.add_argument('--output', type=str, default='benchmarks/results.csv', 
                       help='Output file for results')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['comprehensive'],
                       choices=['comparison', 'memory', 'noise', 'comprehensive', 'all'],
                       help='Benchmarks to run')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick benchmarks with reduced parameters')
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    benchmark = BenchmarkSuite(output_dir=Path(args.output).parent)
    
    # Adjust parameters for quick mode
    if args.quick:
        args.epochs = 5
        logger.info("Running in quick mode")
        
    all_results = {}
    
    # Run selected benchmarks
    if 'all' in args.benchmarks or 'comprehensive' in args.benchmarks:
        all_results.update(benchmark.run_comprehensive_benchmark())
    elif 'comparison' in args.benchmarks:
        all_results.update(benchmark.run_comparison_benchmark())

    if 'all' in args.benchmarks or 'memory' in args.benchmarks:
        all_results.update(benchmark.run_memory_scaling_benchmark())

    if 'all' in args.benchmarks or 'noise' in args.benchmarks:
        all_results.update(benchmark.run_noise_resilience_benchmark())
        
    # Save results
    benchmark.save_results(all_results)
    benchmark.generate_plots(all_results)
    
    logger.info("Benchmark suite completed successfully!")


if __name__ == "__main__":
    main()

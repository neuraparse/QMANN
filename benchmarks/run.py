#!/usr/bin/env python3
"""
QMNN Benchmark Suite

This script runs comprehensive benchmarks comparing QMNN with classical approaches.
"""

import argparse
import time
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import psutil
import gc

from qmnn.models import QMNN, QuantumNeuralNetwork
from qmnn.training import QMNNTrainer
from qmnn.utils import plot_training_history, create_benchmark_plot


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClassicalLSTM(nn.Module):
    """Classical LSTM baseline for comparison."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


class ClassicalTransformer(nn.Module):
    """Classical Transformer baseline for comparison."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        return self.output_projection(x)


class BenchmarkSuite:
    """Comprehensive benchmark suite for QMNN."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.results = []
        
    def generate_synthetic_data(
        self, 
        n_samples: int, 
        seq_len: int, 
        input_dim: int, 
        output_dim: int,
        task_type: str = "classification"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data for benchmarking."""
        
        # Generate input sequences
        X = torch.randn(n_samples, seq_len, input_dim)
        
        if task_type == "classification":
            # Generate classification targets
            y = torch.randint(0, output_dim, (n_samples, seq_len))
        else:
            # Generate regression targets
            y = torch.randn(n_samples, seq_len, output_dim)
            
        return X, y
        
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """Benchmark a single model."""
        
        logger.info(f"Benchmarking {model_name}...")
        
        model = model.to(self.device)
        
        # Setup training
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training metrics
        train_losses = []
        train_accuracies = []
        train_times = []
        memory_usage = []
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            model.train()
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, QMNN):
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
                    
            # Record metrics
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if task_type == "classification":
                accuracy = epoch_correct / epoch_total
                train_accuracies.append(accuracy)
            else:
                train_accuracies.append(1.0 - avg_loss)  # Pseudo-accuracy for regression
                
            epoch_time = time.time() - epoch_start
            train_times.append(epoch_time)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            else:
                memory_usage.append(psutil.Process().memory_info().rss / 1024**2)  # MB
                
        total_train_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        eval_start = time.time()
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if isinstance(model, QMNN):
                    outputs, _ = model(data, store_memory=False)
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
                test_loss += loss.item()
                
                if task_type == "classification":
                    predicted = outputs.argmax(dim=1)
                    test_correct += (predicted == targets).sum().item()
                    test_total += targets.size(0)
                    
        eval_time = time.time() - eval_start
        
        # Calculate final metrics
        avg_test_loss = test_loss / len(test_loader)
        
        if task_type == "classification":
            test_accuracy = test_correct / test_total
        else:
            test_accuracy = 1.0 - avg_test_loss
            
        # Model size
        model_size = sum(p.numel() for p in model.parameters())
        
        # Memory efficiency (for QMNN)
        memory_efficiency = 0.0
        if isinstance(model, QMNN):
            memory_efficiency = model.memory_usage()
            
        results = {
            'model': model_name,
            'train_time': total_train_time,
            'eval_time': eval_time,
            'train_loss': train_losses[-1],
            'test_loss': avg_test_loss,
            'train_accuracy': train_accuracies[-1],
            'test_accuracy': test_accuracy,
            'model_size': model_size,
            'peak_memory_mb': max(memory_usage) if memory_usage else 0,
            'memory_efficiency': memory_efficiency,
            'convergence_epoch': len(train_losses),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'train_times': train_times,
            'memory_usage': memory_usage,
        }
        
        logger.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Train Time: {total_train_time:.2f}s")
        
        return results
        
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
            
            # Create QMNN with specific memory capacity
            model = QMNN(
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
            
            # Create QMNN
            model = QMNN(
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
                model, f"QMNN_noise_{noise_level}", 
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
            'QMNN': QMNN(
                input_dim=10,
                hidden_dim=64,
                output_dim=5,
                memory_capacity=128,
                memory_embedding_dim=32,
            ),
        }

        # Add 2025 advanced models if available
        try:
            from qmnn.quantum_transformers import QuantumTransformerBlock
            models['QuantumTransformer'] = QuantumTransformerBlock(
                d_model=64, n_heads=8, d_ff=256, n_qubits=6
            )
        except ImportError:
            pass

        try:
            from qmnn.error_correction import SurfaceCodeQRAM
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
            plt.title('QMNN Noise Resilience')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "noise_resilience.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Plots saved to {self.output_dir}")


def main():
    """Main benchmark execution."""
    
    parser = argparse.ArgumentParser(description='QMNN Benchmark Suite')
    parser.add_argument('--output', type=str, default='benchmarks/results.csv', 
                       help='Output file for results')
    parser.add_argument('--benchmarks', type=str, nargs='+', 
                       default=['comparison', 'memory', 'noise'],
                       choices=['comparison', 'memory', 'noise', 'all'],
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
    if 'all' in args.benchmarks or 'comparison' in args.benchmarks:
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

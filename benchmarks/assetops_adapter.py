"""
AssetOps Benchmark Adapter for QMNN

This module adapts AssetOps benchmark data for QMNN evaluation,
enabling Industry 4.0 scenario testing and validation.
"""

import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class AssetOpsTask:
    """AssetOps benchmark task definition."""
    task_id: str
    task_type: str  # "maintenance", "optimization", "prediction", "anomaly"
    description: str
    input_format: str  # "time_series", "tabular", "mixed"
    output_format: str  # "classification", "regression", "sequence"
    difficulty: str  # "easy", "medium", "hard"
    industry_domain: str  # "manufacturing", "energy", "logistics", "healthcare"
    memory_requirement: int  # Required memory capacity
    sequence_length: int
    feature_dimension: int
    num_classes: Optional[int] = None


class AssetOpsBenchmarkAdapter:
    """
    Adapter for AssetOps benchmark suite to work with QMNN.
    
    Converts AssetOps JSON data format to PyTorch tensors suitable
    for quantum memory-augmented neural networks.
    """
    
    def __init__(self, benchmark_path: str = "data/assetops/"):
        """
        Initialize AssetOps adapter.
        
        Args:
            benchmark_path: Path to AssetOps benchmark data
        """
        self.benchmark_path = Path(benchmark_path)
        self.tasks = {}
        self.loaded_datasets = {}
        
        # Create benchmark directory if it doesn't exist
        self.benchmark_path.mkdir(parents=True, exist_ok=True)
        
        # Load task definitions
        self._load_task_definitions()
        
    def _load_task_definitions(self):
        """Load AssetOps task definitions."""
        # Define standard AssetOps tasks for Industry 4.0
        self.tasks = {
            "predictive_maintenance": AssetOpsTask(
                task_id="predictive_maintenance",
                task_type="prediction",
                description="Predict equipment failure based on sensor data",
                input_format="time_series",
                output_format="classification",
                difficulty="medium",
                industry_domain="manufacturing",
                memory_requirement=64,
                sequence_length=100,
                feature_dimension=12,
                num_classes=3  # normal, warning, critical
            ),
            "energy_optimization": AssetOpsTask(
                task_id="energy_optimization",
                task_type="optimization",
                description="Optimize energy consumption in smart grid",
                input_format="time_series",
                output_format="regression",
                difficulty="hard",
                industry_domain="energy",
                memory_requirement=128,
                sequence_length=144,  # 24 hours * 6 (10-min intervals)
                feature_dimension=8,
                num_classes=None
            ),
            "supply_chain_anomaly": AssetOpsTask(
                task_id="supply_chain_anomaly",
                task_type="anomaly",
                description="Detect anomalies in supply chain logistics",
                input_format="mixed",
                output_format="classification",
                difficulty="hard",
                industry_domain="logistics",
                memory_requirement=96,
                sequence_length=50,
                feature_dimension=16,
                num_classes=2  # normal, anomaly
            ),
            "quality_control": AssetOpsTask(
                task_id="quality_control",
                task_type="classification",
                description="Automated quality control in manufacturing",
                input_format="tabular",
                output_format="classification",
                difficulty="easy",
                industry_domain="manufacturing",
                memory_requirement=32,
                sequence_length=20,
                feature_dimension=10,
                num_classes=4  # excellent, good, fair, poor
            ),
            "patient_monitoring": AssetOpsTask(
                task_id="patient_monitoring",
                task_type="prediction",
                description="Continuous patient health monitoring",
                input_format="time_series",
                output_format="classification",
                difficulty="medium",
                industry_domain="healthcare",
                memory_requirement=80,
                sequence_length=200,  # Long-term monitoring
                feature_dimension=6,
                num_classes=5  # health status levels
            )
        }
        
    def generate_synthetic_dataset(self, task_id: str, n_samples: int = 1000,
                                 noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic dataset for AssetOps task.
        
        Args:
            task_id: Task identifier
            n_samples: Number of samples to generate
            noise_level: Noise level for synthetic data
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
            
        task = self.tasks[task_id]
        
        # Generate input data based on task characteristics
        if task.input_format == "time_series":
            X = self._generate_time_series_data(task, n_samples, noise_level)
        elif task.input_format == "tabular":
            X = self._generate_tabular_data(task, n_samples, noise_level)
        elif task.input_format == "mixed":
            X = self._generate_mixed_data(task, n_samples, noise_level)
        else:
            raise ValueError(f"Unknown input format: {task.input_format}")
            
        # Generate targets based on output format
        if task.output_format == "classification":
            y = self._generate_classification_targets(task, n_samples, X)
        elif task.output_format == "regression":
            y = self._generate_regression_targets(task, n_samples, X)
        elif task.output_format == "sequence":
            y = self._generate_sequence_targets(task, n_samples, X)
        else:
            raise ValueError(f"Unknown output format: {task.output_format}")
            
        return X, y
        
    def _generate_time_series_data(self, task: AssetOpsTask, n_samples: int, 
                                  noise_level: float) -> torch.Tensor:
        """Generate synthetic time series data."""
        # Create realistic time series patterns based on industry domain
        if task.industry_domain == "manufacturing":
            # Manufacturing sensor data with periodic patterns
            t = torch.linspace(0, 10, task.sequence_length)
            base_pattern = torch.sin(2 * np.pi * t) + 0.5 * torch.sin(4 * np.pi * t)
            
        elif task.industry_domain == "energy":
            # Energy consumption with daily/weekly patterns
            t = torch.linspace(0, 24, task.sequence_length)  # 24 hours
            base_pattern = (torch.sin(2 * np.pi * t / 24) + 
                          0.3 * torch.sin(2 * np.pi * t / (24/7)))  # Daily + weekly
            
        elif task.industry_domain == "healthcare":
            # Physiological signals (heart rate, etc.)
            t = torch.linspace(0, 5, task.sequence_length)
            base_pattern = torch.sin(2 * np.pi * t) + 0.2 * torch.randn(task.sequence_length)
            
        else:
            # Generic pattern
            t = torch.linspace(0, 1, task.sequence_length)
            base_pattern = torch.sin(2 * np.pi * t)
            
        # Replicate pattern across features and samples
        X = base_pattern.unsqueeze(0).unsqueeze(-1).repeat(n_samples, 1, task.feature_dimension)
        
        # Add feature-specific variations
        for i in range(task.feature_dimension):
            phase_shift = 2 * np.pi * i / task.feature_dimension
            amplitude = 0.5 + 0.5 * i / task.feature_dimension
            X[:, :, i] = amplitude * torch.sin(2 * np.pi * t + phase_shift)
            
        # Add noise
        X += noise_level * torch.randn_like(X)
        
        return X
        
    def _generate_tabular_data(self, task: AssetOpsTask, n_samples: int,
                              noise_level: float) -> torch.Tensor:
        """Generate synthetic tabular data."""
        # Create correlated features for tabular data
        X = torch.randn(n_samples, task.sequence_length, task.feature_dimension)
        
        # Add correlations based on industry domain
        if task.industry_domain == "manufacturing":
            # Manufacturing parameters are often correlated
            for i in range(1, task.feature_dimension):
                X[:, :, i] = 0.7 * X[:, :, 0] + 0.3 * X[:, :, i]
                
        # Add noise
        X += noise_level * torch.randn_like(X)
        
        return X
        
    def _generate_mixed_data(self, task: AssetOpsTask, n_samples: int,
                            noise_level: float) -> torch.Tensor:
        """Generate mixed format data (combination of time series and tabular)."""
        # Half time series, half tabular features
        ts_features = task.feature_dimension // 2
        tab_features = task.feature_dimension - ts_features
        
        # Time series part
        X_ts = self._generate_time_series_data(
            AssetOpsTask(
                task_id=task.task_id + "_ts",
                task_type=task.task_type,
                description=task.description,
                input_format="time_series",
                output_format=task.output_format,
                difficulty=task.difficulty,
                industry_domain=task.industry_domain,
                memory_requirement=task.memory_requirement,
                sequence_length=task.sequence_length,
                feature_dimension=ts_features,
                num_classes=task.num_classes
            ),
            n_samples, noise_level
        )
        
        # Tabular part (repeated across sequence)
        X_tab = torch.randn(n_samples, 1, tab_features).repeat(1, task.sequence_length, 1)
        X_tab += noise_level * torch.randn_like(X_tab)
        
        # Concatenate
        X = torch.cat([X_ts, X_tab], dim=-1)
        
        return X
        
    def _generate_classification_targets(self, task: AssetOpsTask, n_samples: int,
                                       X: torch.Tensor) -> torch.Tensor:
        """Generate classification targets based on input patterns."""
        if task.num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks")
            
        # Generate targets based on input characteristics
        if task.task_type == "prediction":
            # Predictive maintenance: based on signal amplitude and trend
            signal_energy = torch.mean(X ** 2, dim=(1, 2))
            signal_trend = torch.mean(X[:, -10:, :] - X[:, :10, :], dim=(1, 2))
            
            # Combine energy and trend for classification
            combined_score = signal_energy + 0.5 * signal_trend
            
        elif task.task_type == "anomaly":
            # Anomaly detection: based on deviation from normal patterns
            mean_signal = torch.mean(X, dim=1, keepdim=True)
            deviation = torch.mean((X - mean_signal) ** 2, dim=(1, 2))
            combined_score = deviation
            
        else:
            # Generic classification based on signal characteristics
            combined_score = torch.mean(X, dim=(1, 2))
            
        # Convert to class labels
        percentiles = torch.quantile(combined_score, 
                                   torch.linspace(0, 1, task.num_classes + 1))
        
        y = torch.zeros(n_samples, task.sequence_length, dtype=torch.long)
        for i in range(task.num_classes):
            mask = (combined_score >= percentiles[i]) & (combined_score < percentiles[i + 1])
            y[mask, :] = i
            
        # Handle edge case for maximum value
        y[combined_score >= percentiles[-1], :] = task.num_classes - 1
        
        return y
        
    def _generate_regression_targets(self, task: AssetOpsTask, n_samples: int,
                                   X: torch.Tensor) -> torch.Tensor:
        """Generate regression targets."""
        if task.task_type == "optimization":
            # Energy optimization: target is optimized consumption
            current_consumption = torch.mean(X, dim=-1)  # [n_samples, seq_len]
            # Optimal consumption is 80% of current with some constraints
            y = 0.8 * current_consumption + 0.1 * torch.randn_like(current_consumption)
            
        else:
            # Generic regression: predict next value
            y = torch.roll(X[:, :, 0], shifts=-1, dims=1)  # Predict next timestep
            y[:, -1] = y[:, -2]  # Handle last timestep
            
        return y
        
    def _generate_sequence_targets(self, task: AssetOpsTask, n_samples: int,
                                 X: torch.Tensor) -> torch.Tensor:
        """Generate sequence-to-sequence targets."""
        # Sequence prediction: predict next sequence
        y = torch.roll(X, shifts=-1, dims=1)
        y[:, -1, :] = y[:, -2, :]  # Handle last timestep
        
        return y
        
    def load_assetops_json(self, json_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load AssetOps data from JSON format.
        
        Args:
            json_path: Path to AssetOps JSON file
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Extract task information
            task_info = data.get('task_info', {})
            task_id = task_info.get('task_id', 'unknown')
            
            # Extract input data
            input_data = data.get('input_data', [])
            target_data = data.get('target_data', [])
            
            # Convert to tensors
            X = torch.tensor(input_data, dtype=torch.float32)
            y = torch.tensor(target_data, dtype=torch.long if 'classification' in task_info.get('output_format', '') else torch.float32)
            
            return X, y
            
        except Exception as e:
            warnings.warn(f"Failed to load AssetOps JSON: {e}")
            # Fallback to synthetic data
            return self.generate_synthetic_dataset('quality_control', 100)
            
    def get_task_info(self, task_id: str) -> AssetOpsTask:
        """Get task information."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
        return self.tasks[task_id]
        
    def list_available_tasks(self) -> List[str]:
        """List all available AssetOps tasks."""
        return list(self.tasks.keys())
        
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get comprehensive benchmark summary."""
        summary = {
            'total_tasks': len(self.tasks),
            'industry_domains': list(set(task.industry_domain for task in self.tasks.values())),
            'task_types': list(set(task.task_type for task in self.tasks.values())),
            'difficulty_levels': list(set(task.difficulty for task in self.tasks.values())),
            'memory_requirements': {
                task_id: task.memory_requirement 
                for task_id, task in self.tasks.items()
            },
            'sequence_lengths': {
                task_id: task.sequence_length 
                for task_id, task in self.tasks.items()
            }
        }
        
        return summary

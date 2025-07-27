"""
Simplified training tests for QMANN.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import warnings
from torch.utils.data import DataLoader, TensorDataset

from qmann.models import QMANN
from qmann.training import QMANNTrainer


class TestQMANNTrainerSimple(unittest.TestCase):
    """Simplified test cases for QMANNTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        self.trainer = QMANNTrainer(
            model=self.model,
            learning_rate=0.001,
            device='cpu'
        )
        
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertEqual(str(self.trainer.device), 'cpu')
        
    def test_training_with_dataloader(self):
        """Test training with DataLoader."""
        # Create dummy dataset
        batch_size = 2
        seq_len = 5
        num_samples = 10
        
        # Generate data
        X = torch.randn(num_samples, seq_len, 4)
        y = torch.randn(num_samples, seq_len, 2)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create criterion
        criterion = nn.MSELoss()
        
        # Train for one epoch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = self.trainer.train_epoch(dataloader, criterion, epoch=0)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('train_loss', metrics)
        self.assertGreater(metrics['train_loss'], 0)
        
    def test_validation(self):
        """Test validation."""
        # Create dummy dataset
        batch_size = 2
        seq_len = 5
        num_samples = 6
        
        # Generate data
        X = torch.randn(num_samples, seq_len, 4)
        y = torch.randn(num_samples, seq_len, 2)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Create criterion
        criterion = nn.MSELoss()
        
        # Validate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = self.trainer.validate(dataloader, criterion)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('val_loss', metrics)
        self.assertGreater(metrics['val_loss'], 0)
        
    def test_full_training_loop(self):
        """Test full training loop."""
        # Create dummy datasets
        batch_size = 2
        seq_len = 3
        num_samples = 8
        
        # Generate data
        X_train = torch.randn(num_samples, seq_len, 4)
        y_train = torch.randn(num_samples, seq_len, 2)
        X_val = torch.randn(num_samples//2, seq_len, 4)
        y_val = torch.randn(num_samples//2, seq_len, 2)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train for a few epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = self.trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                criterion=nn.MSELoss()
            )
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_losses', history)
        self.assertIn('val_losses', history)
        self.assertEqual(len(history['train_losses']), 2)
        self.assertEqual(len(history['val_losses']), 2)
        
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        try:
            # Save checkpoint
            metrics = {'loss': 0.5, 'accuracy': 0.8}
            self.trainer.save_checkpoint(temp_path, epoch=1, metrics=metrics)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load checkpoint
            checkpoint = self.trainer.load_checkpoint(temp_path)
            
            self.assertIsInstance(checkpoint, dict)
            self.assertIn('model_state_dict', checkpoint)
            self.assertIn('optimizer_state_dict', checkpoint)
            self.assertIn('epoch', checkpoint)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTrainingCompatibility(unittest.TestCase):
    """Test training compatibility across environments."""
    
    def test_cpu_training(self):
        """Test training on CPU."""
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        trainer = QMANNTrainer(model=model, device='cpu')
        
        # Create simple data
        X = torch.randn(4, 3, 4)
        y = torch.randn(4, 3, 2)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Test training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = trainer.train_epoch(dataloader, nn.MSELoss(), epoch=0)
        
        self.assertIsInstance(metrics, dict)
        
    def test_gpu_training_compatibility(self):
        """Test GPU training compatibility (if available)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        trainer = QMANNTrainer(model=model, device='cuda')
        
        # Create simple data
        X = torch.randn(4, 3, 4)
        y = torch.randn(4, 3, 2)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Test training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = trainer.train_epoch(dataloader, nn.MSELoss(), epoch=0)
        
        self.assertIsInstance(metrics, dict)


class TestTrainingMetrics(unittest.TestCase):
    """Test training metrics and monitoring."""
    
    def test_metrics_tracking(self):
        """Test metrics tracking during training."""
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        trainer = QMANNTrainer(model=model, device='cpu')
        
        # Initial metrics should be empty
        self.assertEqual(len(trainer.train_losses), 0)
        self.assertEqual(len(trainer.val_losses), 0)
        
        # Create data and train
        X = torch.randn(4, 3, 4)
        y = torch.randn(4, 3, 2)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train_epoch(dataloader, nn.MSELoss(), epoch=0)
        
        # Should have recorded training loss
        self.assertEqual(len(trainer.train_losses), 1)
        self.assertGreater(trainer.train_losses[0], 0)
        
    def test_memory_usage_tracking(self):
        """Test quantum memory usage tracking during training."""
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        trainer = QMANNTrainer(model=model, device='cpu')
        
        # Get initial memory usage
        initial_usage = model.quantum_memory.memory_usage()
        
        # Create data and train
        X = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 2)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train_epoch(dataloader, nn.MSELoss(), epoch=0)
        
        # Get final memory usage
        final_usage = model.quantum_memory.memory_usage()
        
        # Memory usage should be trackable
        self.assertIsInstance(initial_usage, float)
        self.assertIsInstance(final_usage, float)
        self.assertGreaterEqual(final_usage, 0.0)
        self.assertLessEqual(final_usage, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

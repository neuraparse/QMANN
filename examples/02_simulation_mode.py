#!/usr/bin/env python3
"""
QMANN Simulation Mode Example

This example demonstrates classical simulation of quantum operations,
providing practical algorithm development and validation.

💻 SIMULATION MODE:
- Purpose: Algorithm development and validation
- Resources: Up to 20 qubits, noise modeling, quantum-inspired operations
- Cost: FREE (requires computational resources)
- Use case: Development, testing, education, reproducible research
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

# QMANN imports
from qmann import QMANN, QMANNTrainer
from qmann.config import SIMULATION_VALIDATION, validate_experimental_setup


def create_memory_task_dataset(n_samples: int = 1000, seq_len: int = 20, 
                              input_dim: int = 8, memory_lag: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create memory-dependent task dataset.
    
    The model must remember information from early in the sequence
    to make correct predictions later.
    """
    print(f"Creating memory task dataset:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Memory lag: {memory_lag}")
    
    X = torch.randn(n_samples, seq_len, input_dim)
    y = torch.zeros(n_samples, seq_len, dtype=torch.long)
    
    for i in range(n_samples):
        # Create memory cue at the beginning
        memory_cue = torch.randint(0, 3, (1,)).item()
        X[i, 0, 0] = memory_cue  # Embed cue in first position, first feature
        
        # Target depends on memory cue and current input
        for t in range(seq_len):
            if t < memory_lag:
                # Early positions: just echo the memory cue
                y[i, t] = memory_cue
            else:
                # Later positions: combine memory cue with current signal
                current_signal = (X[i, t, 1] > 0).long().item()
                y[i, t] = (memory_cue + current_signal) % 3
                
    return X, y


def run_simulation_experiment():
    """Run comprehensive simulation experiment."""
    print("💻 SIMULATION EXPERIMENT")
    print("=" * 50)
    
    # Validate simulation setup
    print("Validating simulation configuration...")
    is_valid = validate_experimental_setup(SIMULATION_VALIDATION)
    print(f"Configuration valid: {is_valid}\n")
    
    # Create dataset
    X_train, y_train = create_memory_task_dataset(800, 15, 8, 5)
    X_test, y_test = create_memory_task_dataset(200, 15, 8, 5)
    
    print(f"Dataset created: {X_train.shape}, {y_train.shape}")
    
    # Create QMANN model (simulation constraints)
    model = QMANN(
        input_dim=8,
        hidden_dim=64,
        output_dim=3,
        memory_capacity=32,  # Limited by simulation
        max_qubits=8,       # Simulation limit
        n_quantum_layers=2,
        use_attention=True,
        dropout=0.1
    )
    
    print(f"\nQMANN Model (Simulation Mode):")
    quantum_info = model.get_quantum_info()
    print(f"  - Total parameters: {quantum_info['total_parameters']:,}")
    print(f"  - Quantum parameters: {quantum_info['quantum_parameters']:,}")
    print(f"  - Classical parameters: {quantum_info['classical_parameters']:,}")
    print(f"  - Quantum ratio: {quantum_info['quantum_ratio']:.2%}")
    print(f"  - Memory capacity: {quantum_info['memory_capacity']}")
    print(f"  - Max qubits: {quantum_info['n_qubits']}")
    
    # Create classical baseline (LSTM)
    class LSTMBaseline(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.classifier = torch.nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.classifier(lstm_out)
    
    lstm_model = LSTMBaseline(8, 64, 3)
    
    print(f"\nLSTM Baseline:")
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"  - Total parameters: {lstm_params:,}")
    
    # Training setup
    qmann_trainer = QMANNTrainer(model, learning_rate=1e-3)
    lstm_trainer = QMANNTrainer(lstm_model, learning_rate=1e-3)
    
    # Training loop
    n_epochs = 25
    batch_size = 32
    
    qmann_losses = []
    qmann_accuracies = []
    lstm_losses = []
    lstm_accuracies = []
    memory_usages = []
    
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # QMANN training
        qmann_epoch_loss = 0
        qmann_epoch_acc = 0
        memory_usage = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # QMANN forward pass
            qmann_output, memory_info = model(batch_X)
            memory_usage += memory_info['memory_usage']
            
            # QMANN training step
            qmann_metrics = qmann_trainer.train_step(batch_X, batch_y)
            qmann_epoch_loss += qmann_metrics['loss']
            qmann_epoch_acc += qmann_metrics['accuracy']
            
            # LSTM training step
            lstm_metrics = lstm_trainer.train_step(batch_X, batch_y)
            
            n_batches += 1
            
        # LSTM epoch training
        lstm_epoch_loss = 0
        lstm_epoch_acc = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            lstm_metrics = lstm_trainer.train_step(batch_X, batch_y)
            lstm_epoch_loss += lstm_metrics['loss']
            lstm_epoch_acc += lstm_metrics['accuracy']
            
        # Record metrics
        qmann_losses.append(qmann_epoch_loss / n_batches)
        qmann_accuracies.append(qmann_epoch_acc / n_batches)
        lstm_losses.append(lstm_epoch_loss / n_batches)
        lstm_accuracies.append(lstm_epoch_acc / n_batches)
        memory_usages.append(memory_usage / n_batches)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f"  QMANN - Loss: {qmann_losses[-1]:.4f}, Acc: {qmann_accuracies[-1]:.4f}")
            print(f"  LSTM - Loss: {lstm_losses[-1]:.4f}, Acc: {lstm_accuracies[-1]:.4f}")
            print(f"  Memory usage: {memory_usages[-1]:.2%}")
            
    # Final evaluation
    qmann_test_metrics = qmann_trainer.validate(X_test, y_test)
    lstm_test_metrics = lstm_trainer.validate(X_test, y_test)
    
    print(f"\nFinal Test Results:")
    print(f"  QMANN Test Accuracy: {qmann_test_metrics['accuracy']:.4f}")
    print(f"  LSTM Test Accuracy: {lstm_test_metrics['accuracy']:.4f}")
    
    improvement = (qmann_test_metrics['accuracy'] - lstm_test_metrics['accuracy']) / lstm_test_metrics['accuracy'] * 100
    print(f"  QMANN Improvement: {improvement:+.2f}%")
    
    return {
        'qmann_losses': qmann_losses,
        'qmann_accuracies': qmann_accuracies,
        'lstm_losses': lstm_losses,
        'lstm_accuracies': lstm_accuracies,
        'memory_usages': memory_usages,
        'qmann_test_acc': qmann_test_metrics['accuracy'],
        'lstm_test_acc': lstm_test_metrics['accuracy'],
        'improvement': improvement,
        'qmann_params': quantum_info['total_parameters'],
        'lstm_params': lstm_params
    }


def analyze_quantum_simulation_effects():
    """Analyze quantum-specific effects in simulation."""
    print("\n🔬 QUANTUM SIMULATION ANALYSIS")
    print("=" * 50)
    
    # Test different qubit counts
    qubit_counts = [4, 6, 8, 10, 12]
    results = {}
    
    print("Testing quantum memory scaling...")
    
    for n_qubits in qubit_counts:
        print(f"\nTesting {n_qubits} qubits...")
        
        # Create model with specific qubit count
        model = QMANN(
            input_dim=8,
            hidden_dim=32,
            output_dim=3,
            memory_capacity=min(32, 2**n_qubits),
            max_qubits=n_qubits,
            n_quantum_layers=1
        )
        
        # Test forward pass timing
        test_input = torch.randn(10, 10, 8)
        
        start_time = time.time()
        with torch.no_grad():
            output, memory_info = model(test_input)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        quantum_info = model.get_quantum_info()
        
        results[n_qubits] = {
            'execution_time': execution_time,
            'memory_capacity': quantum_info['memory_capacity'],
            'memory_usage': memory_info['memory_usage'],
            'quantum_params': quantum_info['quantum_parameters'],
            'total_params': quantum_info['total_parameters']
        }
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Memory capacity: {quantum_info['memory_capacity']}")
        print(f"  Memory usage: {memory_info['memory_usage']:.2%}")
        
    return results


def plot_simulation_results(training_results: Dict, scaling_results: Dict):
    """Plot simulation results."""
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 3, 1)
    epochs = range(1, len(training_results['qmann_losses']) + 1)
    plt.plot(epochs, training_results['qmann_losses'], 'b-', label='QMANN', linewidth=2)
    plt.plot(epochs, training_results['lstm_losses'], 'r--', label='LSTM', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, training_results['qmann_accuracies'], 'b-', label='QMANN', linewidth=2)
    plt.plot(epochs, training_results['lstm_accuracies'], 'r--', label='LSTM', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, training_results['memory_usages'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Memory Usage')
    plt.title('Quantum Memory Usage')
    plt.grid(True)
    
    # Scaling analysis
    qubit_counts = list(scaling_results.keys())
    execution_times = [scaling_results[q]['execution_time'] for q in qubit_counts]
    memory_capacities = [scaling_results[q]['memory_capacity'] for q in qubit_counts]
    
    plt.subplot(2, 3, 4)
    plt.plot(qubit_counts, execution_times, 'bo-', linewidth=2)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Execution Time (s)')
    plt.title('Simulation Scaling')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.semilogy(qubit_counts, memory_capacities, 'ro-', linewidth=2)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Memory Capacity')
    plt.title('Memory Capacity Scaling')
    plt.grid(True)
    
    # Parameter comparison
    plt.subplot(2, 3, 6)
    models = ['QMANN', 'LSTM']
    params = [training_results['qmann_params'], training_results['lstm_params']]
    colors = ['blue', 'red']
    
    bars = plt.bar(models, params, color=colors, alpha=0.7)
    plt.ylabel('Number of Parameters')
    plt.title('Model Complexity')
    
    # Add value labels on bars
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                f'{param:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_simulation_report(training_results: Dict, scaling_results: Dict) -> str:
    """Generate simulation experiment report."""
    
    report = []
    report.append("QMANN SIMULATION MODE REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"• QMANN vs LSTM improvement: {training_results['improvement']:+.2f}%")
    report.append(f"• QMANN test accuracy: {training_results['qmann_test_acc']:.4f}")
    report.append(f"• LSTM test accuracy: {training_results['lstm_test_acc']:.4f}")
    report.append(f"• Parameter efficiency: {training_results['qmann_params']:,} vs {training_results['lstm_params']:,}")
    report.append("")
    
    # Simulation Details
    report.append("SIMULATION DETAILS")
    report.append("-" * 20)
    report.append("• Mode: Classical simulation of quantum operations")
    report.append("• Max qubits tested: 12 (simulation limit)")
    report.append("• Memory encoding: Amplitude encoding simulation")
    report.append("• Noise modeling: Included in quantum operations")
    report.append("")
    
    # Scaling Analysis
    report.append("SCALING ANALYSIS")
    report.append("-" * 18)
    max_qubits = max(scaling_results.keys())
    max_capacity = scaling_results[max_qubits]['memory_capacity']
    
    report.append(f"• Maximum qubits simulated: {max_qubits}")
    report.append(f"• Maximum memory capacity: {max_capacity:,}")
    report.append(f"• Simulation overhead: Exponential with qubit count")
    report.append(f"• Practical limit: ~20 qubits on standard hardware")
    report.append("")
    
    # Limitations
    report.append("SIMULATION LIMITATIONS")
    report.append("-" * 25)
    report.append("• Classical simulation cannot capture true quantum effects")
    report.append("• Exponential scaling limits qubit count")
    report.append("• Noise models approximate real hardware")
    report.append("• No true quantum advantage demonstrated")
    
    return "\n".join(report)


def main():
    """Main simulation workflow."""
    print("💻 QMANN SIMULATION MODE ANALYSIS")
    print("=" * 50)
    print("Running classical simulation of quantum operations...")
    print("(Free computational experiment)")
    print()
    
    try:
        # Run simulation experiment
        training_results = run_simulation_experiment()
        
        # Analyze scaling
        scaling_results = analyze_quantum_simulation_effects()
        
        # Plot results
        plot_simulation_results(training_results, scaling_results)
        
        # Generate report
        report = generate_simulation_report(training_results, scaling_results)
        
        print("\n" + "=" * 50)
        print("SIMULATION ANALYSIS COMPLETE")
        print("=" * 50)
        print(report)
        
        # Save results
        with open('qmann_simulation_report.txt', 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: qmann_simulation_report.txt")
        
        results = {
            'training': training_results,
            'scaling': scaling_results
        }
        torch.save(results, 'simulation_results.pt')
        print(f"📊 Results saved to: simulation_results.pt")
        
    except Exception as e:
        print(f"❌ Simulation experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

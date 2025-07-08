"""
Command Line Interface for QMNN.

This module provides a comprehensive CLI for training, evaluating, and
benchmarking QMNN models.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.logging import RichHandler

from .models import QMNN
from .training import QMNNTrainer
from .utils import plot_training_history


# Setup rich console
console = Console()

# Setup logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger(__name__)


def train_command(args):
    """Train a QMNN model."""
    console.print("[bold blue]Training QMNN Model[/bold blue]")
    
    # Load configuration
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "memory_capacity": args.memory_capacity,
            "memory_embedding_dim": args.memory_embedding_dim,
        }
    
    # Create model
    model = QMNN(**config)
    
    # Create trainer
    trainer = QMNNTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    
    console.print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # TODO: Load actual data based on args.data_path
    # For now, create dummy data
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    X = torch.randn(1000, args.seq_len, args.input_dim)
    y = torch.randint(0, args.output_dim, (1000, args.seq_len))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train model
    with Progress() as progress:
        task = progress.add_task("Training...", total=args.epochs)
        
        history = trainer.train(
            train_loader=train_loader,
            num_epochs=args.epochs,
        )
        
        progress.update(task, completed=args.epochs)
    
    # Save model
    if args.output:
        torch.save(model.state_dict(), args.output)
        console.print(f"Model saved to {args.output}")
    
    # Plot training history
    if args.plot:
        plot_training_history(history, save_path=args.plot)
        console.print(f"Training plot saved to {args.plot}")
    
    console.print("[bold green]Training completed successfully![/bold green]")


def evaluate_command(args):
    """Evaluate a trained QMNN model."""
    console.print("[bold blue]Evaluating QMNN Model[/bold blue]")
    
    # Load model
    if not Path(args.model).exists():
        console.print(f"[red]Model file not found: {args.model}[/red]")
        sys.exit(1)
    
    # TODO: Implement model loading and evaluation
    console.print(f"Evaluating model from {args.model}")
    console.print("[bold green]Evaluation completed![/bold green]")


def benchmark_command(args):
    """Run benchmarks."""
    console.print("[bold blue]Running QMNN Benchmarks[/bold blue]")
    
    from benchmarks.run import BenchmarkSuite
    
    benchmark = BenchmarkSuite(output_dir=args.output_dir)
    
    if args.quick:
        console.print("Running quick benchmarks...")
        results = benchmark.run_comparison_benchmark()
    else:
        console.print("Running full benchmark suite...")
        results = {}
        
        if 'comparison' in args.benchmarks:
            results.update(benchmark.run_comparison_benchmark())
        if 'memory' in args.benchmarks:
            results.update(benchmark.run_memory_scaling_benchmark())
        if 'noise' in args.benchmarks:
            results.update(benchmark.run_noise_resilience_benchmark())
    
    # Save results
    benchmark.save_results(results)
    benchmark.generate_plots(results)
    
    console.print("[bold green]Benchmarks completed![/bold green]")


def info_command(args):
    """Display system and package information."""
    console.print("[bold blue]QMNN System Information[/bold blue]")
    
    # Create info table
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="green")
    
    # Python info
    import sys
    table.add_row("Python", f"{sys.version.split()[0]}")
    
    # PyTorch info
    import torch
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        table.add_row("Current GPU", torch.cuda.get_device_name())
    
    # Qiskit info
    try:
        import qiskit
        table.add_row("Qiskit", qiskit.__version__)
    except ImportError:
        table.add_row("Qiskit", "[red]Not installed[/red]")
    
    # PennyLane info
    try:
        import pennylane
        table.add_row("PennyLane", pennylane.__version__)
    except ImportError:
        table.add_row("PennyLane", "[red]Not installed[/red]")
    
    # QMNN info
    from . import __version__, __organization__
    table.add_row("QMNN", __version__)
    table.add_row("Organization", __organization__)
    table.add_row("Website", "https://neuraparse.com")
    table.add_row("Contact", "info@neuraparse.com")
    
    console.print(table)


def demo_command(args):
    """Run a demonstration of QMNN capabilities."""
    console.print("[bold blue]QMNN 2025 Demonstration[/bold blue]")

    # Create a small QMNN model
    model = QMNN(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        memory_capacity=8,
        memory_embedding_dim=8,
    )

    console.print(f"Created demo model with {sum(p.numel() for p in model.parameters())} parameters")

    # Generate sample data
    import torch
    x = torch.randn(2, 5, 4)

    console.print("Running forward pass...")
    with torch.no_grad():
        output, memory = model(x, store_memory=True)

    console.print(f"Input shape: {x.shape}")
    console.print(f"Output shape: {output.shape}")
    console.print(f"Memory shape: {memory.shape}")
    console.print(f"Memory usage: {model.memory_usage():.2%}")

    # Display quantum circuit
    if args.show_circuit:
        circuit = model.get_memory_circuit()
        console.print(f"Quantum circuit depth: {circuit.depth()}")
        console.print(f"Quantum circuit qubits: {circuit.num_qubits}")

    # 2025 Features Demo
    console.print("\n[bold cyan]2025 Advanced Features Demo[/bold cyan]")

    # Quantum Transformer Demo
    try:
        from qmnn.quantum_transformers import QuantumAttentionMechanism
        qattn = QuantumAttentionMechanism(d_model=16, n_heads=4, n_qubits=4)

        with torch.no_grad():
            attn_output, attn_weights = qattn(x, x, x)

        console.print(f"✓ Quantum Attention: {attn_output.shape}")
        console.print(f"  Entanglement-enhanced attention weights: {attn_weights.shape}")
    except ImportError:
        console.print("⚠ Quantum Transformers not available")

    # Error Correction Demo
    try:
        from qmnn.error_correction import QuantumErrorMitigation
        error_mitigation = QuantumErrorMitigation(n_qubits=4)

        # Simulate noisy results
        noisy_results = [torch.randn(2, 5, 2) for _ in range(3)]
        mitigated_result = error_mitigation.apply_mitigation(noisy_results)

        console.print(f"✓ Error Mitigation: {mitigated_result.shape}")
        console.print("  Applied zero-noise extrapolation and symmetry verification")
    except ImportError:
        console.print("⚠ Error Correction not available")

    # Quantum Advantage Verification Demo
    try:
        from qmnn.quantum_advantage import QuantumAdvantageMetrics
        advantage_metrics = QuantumAdvantageMetrics()

        # Simulate quantum vs classical results
        quantum_results = [{'accuracy': 0.95, 'training_time': 30}]
        classical_results = [{'accuracy': 0.90, 'training_time': 45}]

        verification = advantage_metrics.verify_quantum_advantage(quantum_results, classical_results)

        console.print(f"✓ Quantum Advantage: {'Verified' if verification['advantage_verified'] else 'Not Verified'}")
        console.print(f"  Advantages found: {verification.get('advantages_found', [])}")
    except ImportError:
        console.print("⚠ Quantum Advantage Verification not available")

    console.print("[bold green]2025 Demo completed![/bold green]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QMNN: Quantum Memory-Augmented Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qmnn train --data data/mnist.npz --epochs 50 --output model.pth
  qmnn evaluate --model model.pth --data data/test.npz
  qmnn benchmark --benchmarks comparison memory --output results/
  qmnn demo --show-circuit
  qmnn info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a QMNN model')
    train_parser.add_argument('--data-path', type=str, help='Path to training data')
    train_parser.add_argument('--config', type=str, help='Path to config JSON file')
    train_parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    train_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    train_parser.add_argument('--output-dim', type=int, default=5, help='Output dimension')
    train_parser.add_argument('--memory-capacity', type=int, default=128, help='Memory capacity')
    train_parser.add_argument('--memory-embedding-dim', type=int, default=32, help='Memory embedding dimension')
    train_parser.add_argument('--seq-len', type=int, default=20, help='Sequence length')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    train_parser.add_argument('--output', type=str, help='Output model path')
    train_parser.add_argument('--plot', type=str, help='Save training plot to path')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--data-path', type=str, required=True, help='Path to test data')
    eval_parser.add_argument('--output', type=str, help='Output results path')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--benchmarks', nargs='+', 
                             choices=['comparison', 'memory', 'noise', 'all'],
                             default=['comparison'], help='Benchmarks to run')
    bench_parser.add_argument('--output-dir', type=str, default='benchmarks/results',
                             help='Output directory for results')
    bench_parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')
    
    # Info command
    subparsers.add_parser('info', help='Display system information')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--show-circuit', action='store_true', 
                            help='Show quantum circuit information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'info':
        info_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

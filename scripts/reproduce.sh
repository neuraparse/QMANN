#!/bin/bash

# QMANN Reproduction Script
# This script reproduces all results from the paper

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="results/reproduction"
DATA_DIR="data"
MODELS_DIR="models"
FIGURES_DIR="paper/figs"

# Create directories
mkdir -p "$RESULTS_DIR" "$DATA_DIR" "$MODELS_DIR" "$FIGURES_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}QMANN Reproduction Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
        print_error "Python 3.9+ required, found $python_version"
        exit 1
    fi
    
    # Check if QMANN is installed
    if ! python -c "import qmann" &> /dev/null; then
        print_warning "QMANN not installed, installing..."
        pip install -e .
    fi
    
    print_status "Dependencies check passed"
}

# Download datasets
download_data() {
    print_status "Downloading datasets..."
    
    cd "$DATA_DIR"
    
    # Download MNIST
    if [ ! -f "mnist.npz" ]; then
        print_status "Downloading MNIST dataset..."
        python -c "
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
import numpy as np
np.savez('mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print('MNIST downloaded successfully')
"
    fi
    
    # Download CIFAR-10
    if [ ! -f "cifar10.npz" ]; then
        print_status "Downloading CIFAR-10 dataset..."
        python -c "
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
import numpy as np
np.savez('cifar10.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print('CIFAR-10 downloaded successfully')
"
    fi
    
    # Verify checksums
    print_status "Verifying data integrity..."
    python ../scripts/verify_data.py
    
    cd ..
    print_status "Data download completed"
}

# Run experiments
run_experiments() {
    print_status "Running experiments..."
    
    # Experiment 1: MNIST Classification
    print_status "Running MNIST classification experiment..."
    python scripts/experiment_mnist.py \
        --output "$RESULTS_DIR/mnist_results.json" \
        --epochs 50 \
        --batch-size 32 \
        --memory-capacity 256
    
    # Experiment 2: Memory Capacity Analysis
    print_status "Running memory capacity analysis..."
    python scripts/experiment_memory_capacity.py \
        --output "$RESULTS_DIR/memory_capacity.json" \
        --max-qubits 10
    
    # Experiment 3: Noise Resilience
    print_status "Running noise resilience experiment..."
    python scripts/experiment_noise.py \
        --output "$RESULTS_DIR/noise_resilience.json" \
        --noise-levels "0.0,0.01,0.05,0.1,0.2"
    
    # Experiment 4: Scaling Analysis
    print_status "Running scaling analysis..."
    python scripts/experiment_scaling.py \
        --output "$RESULTS_DIR/scaling_analysis.json" \
        --problem-sizes "100,500,1000,5000,10000"
    
    # Experiment 5: Comparison with Classical Methods
    print_status "Running comparison with classical methods..."
    python scripts/experiment_comparison.py \
        --output "$RESULTS_DIR/classical_comparison.json" \
        --methods "lstm,transformer,ntm,dnc,qmann"
    
    print_status "All experiments completed"
}

# Generate figures
generate_figures() {
    print_status "Generating figures..."
    
    # Figure 1: Architecture diagram
    print_status "Generating architecture diagram..."
    python scripts/plot_architecture.py \
        --output "$FIGURES_DIR/architecture.pdf"
    
    # Figure 2: Memory capacity comparison
    print_status "Generating memory capacity plot..."
    python scripts/plot_memory_capacity.py \
        --input "$RESULTS_DIR/memory_capacity.json" \
        --output "$FIGURES_DIR/memory_capacity.pdf"
    
    # Figure 3: MNIST results
    print_status "Generating MNIST results plot..."
    python scripts/plot_mnist_results.py \
        --input "$RESULTS_DIR/mnist_results.json" \
        --output "$FIGURES_DIR/mnist_results.pdf"
    
    # Figure 4: Noise resilience
    print_status "Generating noise resilience plot..."
    python scripts/plot_noise_resilience.py \
        --input "$RESULTS_DIR/noise_resilience.json" \
        --output "$FIGURES_DIR/noise_resilience.pdf"
    
    # Figure 5: Scaling analysis
    print_status "Generating scaling analysis plot..."
    python scripts/plot_scaling.py \
        --input "$RESULTS_DIR/scaling_analysis.json" \
        --output "$FIGURES_DIR/scaling_analysis.pdf"
    
    # Figure 6: Classical comparison
    print_status "Generating classical comparison plot..."
    python scripts/plot_comparison.py \
        --input "$RESULTS_DIR/classical_comparison.json" \
        --output "$FIGURES_DIR/classical_comparison.pdf"
    
    print_status "Figure generation completed"
}

# Validate results
validate_results() {
    print_status "Validating results against published values..."
    
    python scripts/validate_results.py \
        --results-dir "$RESULTS_DIR" \
        --tolerance 0.05 \
        --output "$RESULTS_DIR/validation_report.json"
    
    if [ $? -eq 0 ]; then
        print_status "Result validation passed"
    else
        print_warning "Some results differ from published values"
    fi
}

# Generate report
generate_report() {
    print_status "Generating reproduction report..."
    
    python scripts/generate_report.py \
        --results-dir "$RESULTS_DIR" \
        --figures-dir "$FIGURES_DIR" \
        --output "$RESULTS_DIR/reproduction_report.html"
    
    print_status "Report generated: $RESULTS_DIR/reproduction_report.html"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up temporary files..."
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    # Parse command line arguments
    SKIP_DATA=false
    SKIP_EXPERIMENTS=false
    SKIP_FIGURES=false
    QUICK_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --skip-experiments)
                SKIP_EXPERIMENTS=true
                shift
                ;;
            --skip-figures)
                SKIP_FIGURES=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-data        Skip data download"
                echo "  --skip-experiments Skip running experiments"
                echo "  --skip-figures     Skip figure generation"
                echo "  --quick            Quick mode (reduced epochs/samples)"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set quick mode parameters
    if [ "$QUICK_MODE" = true ]; then
        export QMANN_QUICK_MODE=1
        print_warning "Running in quick mode - results may differ from paper"
    fi
    
    # Execute pipeline
    check_dependencies
    
    if [ "$SKIP_DATA" = false ]; then
        download_data
    fi
    
    if [ "$SKIP_EXPERIMENTS" = false ]; then
        run_experiments
    fi
    
    if [ "$SKIP_FIGURES" = false ]; then
        generate_figures
    fi
    
    validate_results
    generate_report
    cleanup
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_status "Reproduction completed successfully!"
    print_status "Total time: $(($duration / 3600))h $(($duration % 3600 / 60))m $(($duration % 60))s"
    print_status "Results available in: $RESULTS_DIR"
    print_status "Figures available in: $FIGURES_DIR"
    print_status "Report available at: $RESULTS_DIR/reproduction_report.html"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"

#!/bin/bash

# QMANN Quick Test Script
# Runs essential tests to verify project setup and production readiness

set -e  # Exit on any error

# Create artifacts directory
mkdir -p artifacts/quicktest

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function to print status
print_status() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"
}

print_info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    print_info "Running test: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        print_status "✓ $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_error "✗ $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Function to check command availability
check_command() {
    local cmd="$1"
    local description="$2"
    
    if command -v "$cmd" &> /dev/null; then
        print_status "✓ $description ($cmd found)"
        return 0
    else
        print_error "✗ $description ($cmd not found)"
        return 1
    fi
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}QMANN Quick Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"

# Check system requirements
print_info "Checking system requirements..."

# Check for python command (Windows uses 'python', Linux uses 'python3')
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    print_status "✓ Python 3 (python3 found)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    print_status "✓ Python 3 (python found)"
else
    print_error "✗ Python 3 (neither python3 nor python found)"
    exit 1
fi
check_command "pip" "pip package manager"
check_command "git" "Git version control"

# Check Python version
print_info "Checking Python version..."
python_version=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
python_check=$($PYTHON_CMD -c "import sys; print(1 if sys.version_info >= (3, 9) else 0)")
if [[ "$python_check" -eq 1 ]]; then
    print_status "✓ Python version: $python_version"
else
    print_error "✗ Python version: $python_version (requires 3.9+)"
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/qmann" ]; then
    print_error "Not in QMANN project root directory"
    exit 1
fi

print_status "✓ In QMANN project directory"

# Check if virtual environment is recommended
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Not in a virtual environment (recommended but not required)"
fi

# Test 1: Package installation
print_info "Testing package installation..."
run_test "Package importable" "$PYTHON_CMD -c 'import qmann'"

# Test 2: Core dependencies
print_info "Testing core dependencies..."
run_test "PyTorch import" "$PYTHON_CMD -c 'import torch'"
run_test "NumPy import" "$PYTHON_CMD -c 'import numpy'"
run_test "Qiskit import" "$PYTHON_CMD -c 'import qiskit'"

# Test 3: QMANN core components
print_info "Testing QMANN core components..."
run_test "QRAM import" "$PYTHON_CMD -c 'from qmann.core import QRAM'"
run_test "QMANN model import" "$PYTHON_CMD -c 'from qmann.models import QMANN'"
run_test "Trainer import" "$PYTHON_CMD -c 'from qmann.training import QMANNTrainer'"

# Test 4: Basic functionality
print_info "Testing basic functionality..."
cat > /tmp/qmann_test.py << 'EOF'
import torch
import numpy as np
from qmann.core import QRAM, QuantumMemory
from qmann.models import QMANN

# Test QRAM creation
qram = QRAM(memory_size=4, address_qubits=2)
print("QRAM created successfully")

# Test QuantumMemory
qmem = QuantumMemory(capacity=8, embedding_dim=4)
print("QuantumMemory created successfully")

# Test QMANN model
model = QMANN(
    input_dim=4,
    hidden_dim=16,
    output_dim=2,
    memory_capacity=8,
    memory_embedding_dim=8
)
print("QMANN model created successfully")

# Test forward pass
x = torch.randn(1, 3, 4)
with torch.no_grad():
    output, memory = model(x)
print(f"Forward pass successful: output shape {output.shape}")
print("All basic functionality tests passed!")
EOF

run_test "Basic functionality" "$PYTHON_CMD /tmp/qmann_test.py"

# Test 5: CLI interface
print_info "Testing CLI interface..."
run_test "CLI help" "$PYTHON_CMD -m qmann.cli --help"
run_test "CLI info command" "$PYTHON_CMD -m qmann.cli info"

# Test 6: Unit tests (if available)
if [ -d "tests" ] && [ -f "tests/test_core.py" ]; then
    print_info "Running unit tests..."
    if command -v pytest &> /dev/null; then
        run_test "Core unit tests" "pytest tests/test_core.py -v --tb=short"
    else
        run_test "Core unit tests (unittest)" "$PYTHON_CMD -m unittest tests.test_core -v"
    fi
else
    print_warning "Unit tests not found, skipping"
fi

# Test 7: Code quality (if tools available)
print_info "Checking code quality..."
if command -v black &> /dev/null; then
    run_test "Code formatting (black)" "black --check src/ --diff"
else
    print_warning "black not found, skipping format check"
fi

if command -v isort &> /dev/null; then
    run_test "Import sorting (isort)" "isort --check-only src/"
else
    print_warning "isort not found, skipping import check"
fi

# Test 8: Documentation
print_info "Checking documentation..."
run_test "README exists" "[ -f README.md ]"
run_test "Paper directory" "[ -d paper ]"
run_test "Paper main.tex" "[ -f paper/main.tex ]"

# Test 9: Docker (if available)
if command -v docker &> /dev/null; then
    print_info "Testing Docker setup..."
    run_test "Dockerfile exists" "[ -f docker/Dockerfile ]"
    run_test "docker-compose.yml exists" "[ -f docker-compose.yml ]"
else
    print_warning "Docker not found, skipping Docker tests"
fi

# Test 10: Paper compilation (if LaTeX available)
if command -v pdflatex &> /dev/null && [ -f "paper/main.tex" ]; then
    print_info "Testing paper compilation..."
    cd paper
    run_test "LaTeX compilation" "pdflatex -interaction=nonstopmode main.tex"
    cd ..
else
    print_warning "LaTeX not found or paper not available, skipping paper test"
fi

# Cleanup
rm -f /tmp/qmann_test.py

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    print_status "All tests passed! ($TESTS_PASSED/$TESTS_TOTAL)"
    echo ""
    print_status "🎉 QMANN setup is working correctly!"
    echo ""
    echo "Next steps:"
    echo "  1. Run full test suite: make test"
    echo "  2. Try the demo: python -m qmann.cli demo"
    echo "  3. Start development: make dev-server"
    echo "  4. Read documentation: open README.md"
    exit 0
else
    print_error "Some tests failed ($TESTS_FAILED/$TESTS_TOTAL failed)"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check Python version (requires 3.9+)"
    echo "  2. Install dependencies: pip install -e ."
    echo "  3. Check virtual environment setup"
    echo "  4. See README.md for detailed setup instructions"
    exit 1
fi

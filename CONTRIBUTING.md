# Contributing to QMNN

Thank you for your interest in contributing to Quantum Memory-Augmented Neural Networks (QMNN)! This project is open source under the Apache 2.0 license, and we welcome contributions from the community.

## 🚀 Welcome Contributors!

We're excited to have you contribute to advancing quantum machine learning research. Whether you're fixing bugs, adding features, improving documentation, or sharing research insights, your contributions help make quantum computing more accessible to everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/QMANN.git
   cd QMANN
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/bayrameker/QMANN.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)

### Local Development

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Docker Development

Alternatively, use Docker for development:

```bash
docker-compose up qmnn-dev
```

This will start a Jupyter Lab environment with all dependencies installed.

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Add new functionality
- **Documentation improvements**: Enhance or fix documentation
- **Performance optimizations**: Improve code efficiency
- **Test additions**: Add or improve test coverage
- **Examples and tutorials**: Create educational content

### Coding Standards

- **Python Style**: Follow PEP 8 guidelines
- **Code Formatting**: Use `black` for code formatting
- **Import Sorting**: Use `isort` for import organization
- **Type Hints**: Include type hints for all functions and methods
- **Docstrings**: Use Google-style docstrings for all public functions

### Code Quality Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run all quality checks:
```bash
make lint
make test
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Add tests** for new functionality

4. **Update documentation** if necessary

5. **Run tests and quality checks**:
   ```bash
   pytest tests/
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add quantum memory optimization"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Title**: Use a descriptive title following conventional commits format
- **Description**: Provide a clear description of changes
- **Tests**: Include tests for new functionality
- **Documentation**: Update relevant documentation
- **Breaking Changes**: Clearly mark any breaking changes

### Commit Message Format

We follow the [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

Examples:
```
feat(core): add quantum memory compression
fix(models): resolve QMNN gradient flow issue
docs(readme): update installation instructions
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, package versions
- **Steps to reproduce**: Clear steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full error messages and stack traces
- **Minimal example**: Minimal code example that reproduces the issue

### Feature Requests

For feature requests, please include:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: Your ideas for implementation
- **Alternatives**: Alternative solutions you've considered
- **Additional context**: Any other relevant information

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch for next release
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fix branches

### Release Process

1. Features are developed in `feature/*` branches
2. Features are merged to `develop` via pull requests
3. Release candidates are created from `develop`
4. Stable releases are merged to `main` and tagged

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/qmnn

# Run specific test file
pytest tests/test_core.py

# Run tests with specific markers
pytest -m "not slow"
```

### Test Categories

- **Unit tests**: Test individual components
- **Integration tests**: Test component interactions
- **Benchmark tests**: Performance and scaling tests
- **Slow tests**: Long-running tests (marked with `@pytest.mark.slow`)

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies
- Use fixtures for common test data

## Documentation

### Types of Documentation

- **API Documentation**: Docstrings in code
- **User Guide**: Usage examples and tutorials
- **Developer Guide**: Development and contribution information
- **Paper Documentation**: LaTeX documentation in `paper/`

### Building Documentation

```bash
# Build API documentation
sphinx-build -b html docs/ docs/_build/

# Build paper
cd paper/
make pdf
```

### Documentation Guidelines

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Test all code examples

## Research Contributions

### Paper Contributions

If contributing to the research paper:

1. Follow LaTeX best practices
2. Ensure all figures are high-quality and properly sized
3. Verify all references are complete and properly formatted
4. Run the arXiv build pipeline to check compliance

### Experimental Results

When contributing experimental results:

1. Ensure reproducibility with provided scripts
2. Include statistical significance tests
3. Document experimental setup thoroughly
4. Provide raw data when possible

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Contact Neura Parse at [info@neuraparse.com](mailto:info@neuraparse.com)
- **Website**: Visit [neuraparse.com](https://neuraparse.com) for more information

## Recognition

Contributors will be acknowledged in:

- GitHub contributors list
- Paper acknowledgments (for significant contributions)
- Release notes
- Project documentation

Thank you for contributing to QMNN! Your contributions help advance the field of quantum machine learning.

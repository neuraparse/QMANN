# Multi-stage build for QMANN
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install -e ".[all]"

# Development stage
FROM base as development

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook \
    pre-commit \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Set up pre-commit hooks
RUN git init . && pre-commit install || true

# Expose ports for Jupyter and MLflow
EXPOSE 8888 5000

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY benchmarks/ ./benchmarks/
COPY data/ ./data/
COPY circuits/ ./circuits/
COPY pyproject.toml ./
COPY README.md ./

# Install package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 qmann && \
    chown -R qmann:qmann /app

USER qmann

# Default command
CMD ["python", "-m", "qmann.cli"]

# Benchmark stage
FROM production as benchmark

USER root

# Install additional benchmarking tools
RUN pip install \
    memory-profiler \
    psutil \
    py-spy \
    line-profiler

# Copy benchmark scripts
COPY benchmarks/ ./benchmarks/
COPY scripts/run_benchmarks.sh ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh

USER qmann

# Default command for benchmarks
CMD ["./scripts/run_benchmarks.sh"]

# Paper compilation stage
FROM ubuntu:22.04 as paper

ENV DEBIAN_FRONTEND=noninteractive

# Install LaTeX and tools
RUN apt-get update && apt-get install -y \
    texlive-full \
    texlive-latex-extra \
    texlive-science \
    texlive-fonts-extra \
    latexmk \
    ghostscript \
    poppler-utils \
    qpdf \
    python3 \
    python3-pip \
    python3-yaml \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /paper

# Copy paper files
COPY paper/ ./

# Default command to build paper
CMD ["make", "pdf"]

# Final multi-target image
FROM production as final

# Add labels for metadata
LABEL org.opencontainers.image.title="QMANN: Quantum Memory-Augmented Neural Networks"
LABEL org.opencontainers.image.description="A quantum computing framework for memory-augmented neural networks"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Neura Parse <info@neuraparse.com>"
LABEL org.opencontainers.image.vendor="Neura Parse"
LABEL org.opencontainers.image.url="https://neuraparse.com"
LABEL org.opencontainers.image.source="https://github.com/neuraparse/QMANN"
LABEL org.opencontainers.image.licenses="Neura Parse Dual License"
LABEL patent.protected="true"
LABEL commercial.license.required="true"
LABEL licensing.contact="info@neuraparse.com"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import qmann; print('QMANN is healthy')" || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "qmann.cli"]

#!/bin/bash
# Development and build script for SQL Lineage Analyzer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show help
show_help() {
    echo "SQL Lineage Analyzer Development Script"
    echo "Usage: ./dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install      Install package"
    echo "  install-dev  Install package with development dependencies"
    echo "  test         Run tests"
    echo "  test-cov     Run tests with coverage"
    echo "  lint         Run linting (flake8)"
    echo "  format       Format code (black + isort)"
    echo "  type-check   Run type checking (mypy)"
    echo "  check        Run all checks (lint + type-check + test)"
    echo "  clean        Clean build artifacts"
    echo "  build        Build package"
    echo "  dev-setup    Setup development environment"
    echo "  dev-check    Run development checks (format + lint + type-check + test)"
    echo "  example      Run example usage script"
    echo "  help         Show this help message"
}

# Install package
install() {
    print_status "Installing package..."
    pip install -e .
    print_success "Package installed successfully"
}

# Install with development dependencies
install_dev() {
    print_status "Installing package with development dependencies..."
    pip install -e ".[dev]"
    print_success "Development environment installed successfully"
}

# Run tests
test() {
    print_status "Running tests..."
    if command -v pytest >/dev/null 2>&1; then
        pytest
        print_success "Tests completed successfully"
    else
        print_error "pytest not found. Run './dev.sh install-dev' first"
        exit 1
    fi
}

# Run tests with coverage
test_cov() {
    print_status "Running tests with coverage..."
    if command -v pytest >/dev/null 2>&1; then
        pytest --cov=src/analyzer --cov-report=html --cov-report=term
        print_success "Tests with coverage completed successfully"
        print_status "Coverage report saved to htmlcov/index.html"
    else
        print_error "pytest not found. Run './dev.sh install-dev' first"
        exit 1
    fi
}

# Run linting
lint() {
    print_status "Running linting..."
    if command -v flake8 >/dev/null 2>&1; then
        flake8 src/ tests/ examples/ || {
            print_error "Linting failed"
            exit 1
        }
        print_success "Linting passed"
    else
        print_error "flake8 not found. Run './dev.sh install-dev' first"
        exit 1
    fi
}

# Format code
format_code() {
    print_status "Formatting code..."
    if command -v black >/dev/null 2>&1 && command -v isort >/dev/null 2>&1; then
        black src/ tests/ examples/
        isort src/ tests/ examples/
        print_success "Code formatted successfully"
    else
        print_error "black or isort not found. Run './dev.sh install-dev' first"
        exit 1
    fi
}

# Run type checking
type_check() {
    print_status "Running type checking..."
    if command -v mypy >/dev/null 2>&1; then
        mypy src/analyzer/ || {
            print_warning "Type checking found issues"
        }
        print_success "Type checking completed"
    else
        print_error "mypy not found. Run './dev.sh install-dev' first"
        exit 1
    fi
}

# Run all checks
check() {
    print_status "Running all checks..."
    lint
    type_check
    test
    print_success "All checks passed!"
}

# Clean build artifacts
clean() {
    print_status "Cleaning build artifacts..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf htmlcov/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_success "Build artifacts cleaned"
}

# Build package
build() {
    print_status "Building package..."
    clean
    if command -v python >/dev/null 2>&1; then
        python -m build
        print_success "Package built successfully"
        print_status "Built packages:"
        ls -la dist/
    else
        print_error "python not found"
        exit 1
    fi
}

# Setup development environment
dev_setup() {
    print_status "Setting up development environment..."
    install_dev
    if command -v pre-commit >/dev/null 2>&1; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi
    print_success "Development environment setup complete"
}

# Run development checks
dev_check() {
    print_status "Running development checks..."
    format_code
    lint
    type_check
    test
    print_success "All development checks passed!"
}

# Run example script
run_example() {
    print_status "Running example usage script..."
    if [ -f "examples/sample_usage.py" ]; then
        python examples/sample_usage.py
        print_success "Example script completed"
    else
        print_error "Example script not found at examples/sample_usage.py"
        exit 1
    fi
}

# Main script logic
case "${1:-help}" in
    install)
        install
        ;;
    install-dev)
        install_dev
        ;;
    test)
        test
        ;;
    test-cov)
        test_cov
        ;;
    lint)
        lint
        ;;
    format)
        format_code
        ;;
    type-check)
        type_check
        ;;
    check)
        check
        ;;
    clean)
        clean
        ;;
    build)
        build
        ;;
    dev-setup)
        dev_setup
        ;;
    dev-check)
        dev_check
        ;;
    example)
        run_example
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
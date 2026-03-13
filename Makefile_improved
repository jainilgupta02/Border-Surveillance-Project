# ========================================
# BORDER SURVEILLANCE AI - MAKEFILE
# ========================================
# Automation for common development tasks
#
# Quick Start:
#   make setup    - Create venv and install dependencies
#   make test     - Run all tests
#   make clean    - Remove venv and cache files
#
# Requirements: Python 3.10 or 3.11
# ========================================

# ========================================
# CONFIGURATION
# ========================================

# Python version (CHANGE THIS if your python3 points to wrong version)
# Options: python3.10, python3.11, python, python3
PYTHON := python3.10

# Project name
PROJECT_NAME := border-surveillance-ai

# Virtual environment directory
VENV := venv

# Python executable in virtual environment
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV)/Scripts/python.exe
    VENV_ACTIVATE := $(VENV)/Scripts/activate
else
    VENV_PYTHON := $(VENV)/bin/python
    VENV_ACTIVATE := $(VENV)/bin/activate
endif

# ========================================
# PHONY TARGETS (not actual files)
# ========================================

.PHONY: help setup install install-dev test test-unit test-integration \
        lint format type-check clean clean-all coverage report \
        run-local check-python verify

# ========================================
# DEFAULT TARGET
# ========================================

help:
	@echo "╔════════════════════════════════════════════════════════╗"
	@echo "║   Border Surveillance AI - Development Commands       ║"
	@echo "╚════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create venv and install all dependencies"
	@echo "  make install        - Install/update dependencies only"
	@echo "  make install-dev    - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests with coverage"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make coverage       - Generate coverage report (HTML)"
	@echo "  make report         - Generate test report (HTML)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Check code style (flake8)"
	@echo "  make format         - Format code (black)"
	@echo "  make type-check     - Type checking (mypy)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove venv"
	@echo "  make clean-all      - Remove venv, cache, and generated files"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-python   - Verify Python version"
	@echo "  make verify         - Verify complete setup"
	@echo ""
	@echo "Current Python: $(PYTHON)"
	@echo ""

# ========================================
# PYTHON VERSION CHECK
# ========================================

check-python:
	@echo "🔍 Checking Python version..."
	@$(PYTHON) --version 2>&1 | grep -q "Python 3.1[0-1]" || \
		(echo "❌ Error: Python 3.10 or 3.11 required!" && \
		 echo "   Your Python: $$($(PYTHON) --version)" && \
		 echo "   Please install Python 3.10 or 3.11" && \
		 echo "   Or update PYTHON variable in Makefile" && exit 1)
	@echo "✅ Python version OK: $$($(PYTHON) --version)"

# ========================================
# SETUP & INSTALLATION
# ========================================

setup: check-python
	@echo "🚀 Setting up Border Surveillance AI project..."
	@echo ""
	@if [ -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment already exists at $(VENV)"; \
		echo "   Run 'make clean' first to recreate it"; \
		exit 1; \
	fi
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created"
	@echo ""
	@echo "⬆️  Upgrading pip..."
	$(VENV_PYTHON) -m pip install --upgrade pip
	@echo ""
	@echo "📥 Installing dependencies (this may take 5-10 minutes)..."
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "📝 Next steps:"
	@echo "   1. Activate virtual environment:"
ifeq ($(OS),Windows_NT)
	@echo "      $(VENV_ACTIVATE)"
else
	@echo "      source $(VENV_ACTIVATE)"
endif
	@echo "   2. Verify setup:"
	@echo "      make verify"
	@echo ""

install:
	@echo "📥 Installing/updating dependencies..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found!"; \
		echo "   Run 'make setup' first"; \
		exit 1; \
	fi
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev:
	@echo "📥 Installing development dependencies..."
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(VENV_PYTHON) -m pip install pytest pytest-cov pytest-mock pytest-benchmark
	$(VENV_PYTHON) -m pip install flake8 black pylint mypy
	$(VENV_PYTHON) -m pip install jupyter notebook ipykernel
	@echo "✅ Development dependencies installed"

# ========================================
# VERIFICATION
# ========================================

verify:
	@echo "🔍 Verifying installation..."
	@echo ""
	@echo "1️⃣  Python version:"
	@$(VENV_PYTHON) --version
	@echo ""
	@echo "2️⃣  OpenCV:"
	@$(VENV_PYTHON) -c "import cv2; print(f'   OpenCV {cv2.__version__}')" || \
		(echo "❌ OpenCV import failed!" && exit 1)
	@echo ""
	@echo "3️⃣  YOLOv8:"
	@$(VENV_PYTHON) -c "from ultralytics import YOLO; print('   YOLOv8 installed')" || \
		(echo "❌ YOLOv8 import failed!" && exit 1)
	@echo ""
	@echo "4️⃣  PyTorch:"
	@$(VENV_PYTHON) -c "import torch; print(f'   PyTorch {torch.__version__}')" || \
		(echo "❌ PyTorch import failed!" && exit 1)
	@echo ""
	@echo "5️⃣  Azure SDK:"
	@$(VENV_PYTHON) -c "from azure.storage.blob import BlobServiceClient; print('   Azure SDK installed')" || \
		(echo "❌ Azure SDK import failed!" && exit 1)
	@echo ""
	@echo "6️⃣  Pandas:"
	@$(VENV_PYTHON) -c "import pandas; print(f'   Pandas {pandas.__version__}')" || \
		(echo "❌ Pandas import failed!" && exit 1)
	@echo ""
	@echo "✅ All dependencies verified successfully!"
	@echo ""

# ========================================
# TESTING
# ========================================

test:
	@echo "🧪 Running all tests with coverage..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found!"; \
		echo "   Run 'make setup' first"; \
		exit 1; \
	fi
	$(VENV_PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing
	@echo ""
	@echo "✅ Tests complete!"

test-unit:
	@echo "🧪 Running unit tests..."
	$(VENV_PYTHON) -m pytest tests/ -v -m "not integration"

test-integration:
	@echo "🧪 Running integration tests..."
	$(VENV_PYTHON) -m pytest tests/ -v -m "integration"

coverage:
	@echo "📊 Generating coverage report..."
	$(VENV_PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo ""
	@echo "✅ Coverage report generated: htmlcov/index.html"
	@echo "   Open in browser to view"

report:
	@echo "📄 Generating test report..."
	$(VENV_PYTHON) -m pytest tests/ --html=reports/test_report.html --self-contained-html
	@echo "✅ Test report generated: reports/test_report.html"

# ========================================
# CODE QUALITY
# ========================================

lint:
	@echo "🔍 Checking code style with flake8..."
	$(VENV_PYTHON) -m flake8 src/ tests/ --max-line-length=100 --exclude=venv
	@echo "✅ Code style check passed"

format:
	@echo "✨ Formatting code with black..."
	$(VENV_PYTHON) -m black src/ tests/ --line-length=100
	@echo "✅ Code formatted"

type-check:
	@echo "🔍 Type checking with mypy..."
	$(VENV_PYTHON) -m mypy src/ --ignore-missing-imports
	@echo "✅ Type check passed"

# ========================================
# CLEANUP
# ========================================

clean:
	@echo "🧹 Cleaning up virtual environment..."
	@if [ -d "$(VENV)" ]; then \
		rm -rf $(VENV); \
		echo "✅ Virtual environment removed"; \
	else \
		echo "⚠️  Virtual environment not found (already clean)"; \
	fi

clean-all: clean
	@echo "🧹 Deep cleaning..."
	@echo "   Removing Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "   Removing pytest cache..."
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "   Removing coverage data..."
	@rm -rf htmlcov .coverage 2>/dev/null || true
	@echo "   Removing test reports..."
	@rm -rf reports 2>/dev/null || true
	@echo "   Removing build artifacts..."
	@rm -rf build dist *.egg-info 2>/dev/null || true
	@echo "✅ Deep clean complete"

# ========================================
# DEVELOPMENT UTILITIES
# ========================================

run-local:
	@echo "🚀 Running preprocessing demo..."
	$(VENV_PYTHON) src/preprocessing.py

jupyter:
	@echo "📓 Starting Jupyter Notebook..."
	$(VENV_PYTHON) -m jupyter notebook

# ========================================
# QUICK COMMANDS
# ========================================

# Shorthand for common commands
s: setup
i: install
t: test
c: clean
v: verify
l: lint
f: format

# ========================================
# TROUBLESHOOTING
# ========================================

.PHONY: doctor
doctor:
	@echo "🏥 Running diagnostics..."
	@echo ""
	@echo "System Information:"
	@echo "==================="
	@uname -a || echo "N/A"
	@echo ""
	@echo "Python Information:"
	@echo "==================="
	@which $(PYTHON) || echo "Python not found at: $(PYTHON)"
	@$(PYTHON) --version || echo "Cannot get Python version"
	@echo ""
	@echo "Virtual Environment:"
	@echo "===================="
	@if [ -d "$(VENV)" ]; then \
		echo "✅ Virtual environment exists at: $(VENV)"; \
		echo "Python: $(VENV_PYTHON)"; \
		$(VENV_PYTHON) --version; \
	else \
		echo "❌ Virtual environment not found"; \
	fi
	@echo ""
	@echo "Pip Information:"
	@echo "================"
	@if [ -d "$(VENV)" ]; then \
		$(VENV_PYTHON) -m pip --version; \
	else \
		echo "Virtual environment needed"; \
	fi
	@echo ""

# ========================================
# NOTES
# ========================================

# To use this Makefile:
# 1. Ensure Python 3.10 or 3.11 is installed
# 2. Update PYTHON variable if needed
# 3. Run: make setup
# 4. Run: make verify
# 5. Start coding!

# Common issues:
# - "python3.10: command not found"
#   → Install Python 3.10 or change PYTHON to python3.11
# - "Virtual environment already exists"
#   → Run 'make clean' first
# - Import errors after setup
#   → Run 'make verify' to diagnose

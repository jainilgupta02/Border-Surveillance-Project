# ========================================
# BORDER SURVEILLANCE AI — MAKEFILE
# ========================================
# Common development and evaluation tasks.
#
# Quick Start (Evaluator):
#   make setup      → create venv + install deps
#   make verify     → confirm all imports work
#   make run        → run pipeline on test video
#   make dashboard  → launch Streamlit dashboard
#   make test       → run full test suite
#
# Requirements: Python 3.10 or 3.11
# ========================================


# ── CONFIGURATION ───────────────────────

# Auto-detect python — tries python3.10, then python3.11, then python3
PYTHON := $(shell python3.10 --version > /dev/null 2>&1 && echo python3.10 || \
                  python3.11 --version > /dev/null 2>&1 && echo python3.11 || \
                  echo python3)

PROJECT_NAME := border-surveillance-ai
VENV         := venv
TEST_VIDEO   := data/test_videos/dota_aerial_test.mp4

ifeq ($(OS),Windows_NT)
    VENV_PYTHON   := $(VENV)/Scripts/python.exe
    VENV_ACTIVATE := $(VENV)/Scripts/activate
else
    VENV_PYTHON   := $(VENV)/bin/python
    VENV_ACTIVATE := $(VENV)/bin/activate
endif


# ── PHONY TARGETS ───────────────────────

.PHONY: help setup install install-dev verify \
        run run-camera dashboard pilot smoke \
        test test-unit coverage report \
        lint format type-check \
        clean clean-all doctor \
        s i t c v l f


# ── DEFAULT: HELP ───────────────────────

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║      Border Surveillance AI — Development Commands       ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  🚀 Setup"
	@echo "     make setup        Create venv and install all dependencies"
	@echo "     make install      Update dependencies in existing venv"
	@echo "     make verify       Verify all imports and models load correctly"
	@echo ""
	@echo "  ▶️  Run"
	@echo "     make run          Run pipeline on test video (recommended)"
	@echo "     make run-camera   Run pipeline on live camera (index 0)"
	@echo "     make dashboard    Launch Streamlit dashboard"
	@echo "     make pilot        Run manual integration checker"
	@echo "     make smoke        Quick smoke test"
	@echo ""
	@echo "  🧪 Testing"
	@echo "     make test         Run all tests with coverage"
	@echo "     make test-unit    Run unit tests only"
	@echo "     make coverage     Generate HTML coverage report"
	@echo "     make report       Generate HTML test report"
	@echo ""
	@echo "  ✨ Code Quality"
	@echo "     make lint         Flake8 style check"
	@echo "     make format       Black auto-format"
	@echo "     make type-check   Mypy type checking"
	@echo ""
	@echo "  🧹 Cleanup"
	@echo "     make clean        Remove venv"
	@echo "     make clean-all    Remove venv, cache, and all generated files"
	@echo ""
	@echo "  🏥 Diagnostics"
	@echo "     make doctor       Full environment diagnostics"
	@echo ""
	@echo "  Detected Python: $(PYTHON)"
	@echo ""


# ── SETUP & INSTALLATION ────────────────

setup:
	@echo ""
	@echo "🚀 Setting up Border Surveillance AI..."
	@echo ""
	@if [ -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment already exists."; \
		echo "   Run 'make clean' first to recreate it."; \
		exit 1; \
	fi
	@echo "📦 Creating virtual environment with $(PYTHON)..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created"
	@echo ""
	@echo "⬆️  Upgrading pip..."
	$(VENV_PYTHON) -m pip install --upgrade pip --quiet
	@echo ""
	@echo "📥 Installing dependencies (this may take 5–10 minutes)..."
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "📝 Next steps:"
	@echo "   1. Copy .env.example → .env and fill in credentials (optional)"
	@echo "   2. Run: make verify"
	@echo "   3. Run: make run"
	@echo ""

install:
	@echo "📥 Installing/updating dependencies..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(VENV_PYTHON) -m pip install --upgrade pip --quiet
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "✅ Dependencies updated"

install-dev:
	@echo "📥 Installing dev dependencies..."
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	@echo "✅ Dev dependencies installed"


# ── VERIFICATION ────────────────────────

verify:
	@echo ""
	@echo "🔍 Verifying installation..."
	@echo ""
	@echo "1️⃣  Python:"
	@$(VENV_PYTHON) --version
	@echo ""
	@echo "2️⃣  OpenCV:"
	@$(VENV_PYTHON) -c "import cv2; print(f'   cv2 {cv2.__version__} ✅')" || \
		(echo "   ❌ OpenCV import failed" && exit 1)
	@echo ""
	@echo "3️⃣  YOLOv8:"
	@$(VENV_PYTHON) -c "from ultralytics import YOLO; print('   YOLOv8 ✅')" || \
		(echo "   ❌ YOLOv8 import failed" && exit 1)
	@echo ""
	@echo "4️⃣  PyTorch:"
	@$(VENV_PYTHON) -c "import torch; print(f'   PyTorch {torch.__version__} ✅')" || \
		(echo "   ❌ PyTorch import failed" && exit 1)
	@echo ""
	@echo "5️⃣  scikit-learn:"
	@$(VENV_PYTHON) -c "import sklearn; print(f'   scikit-learn {sklearn.__version__} ✅')" || \
		(echo "   ❌ scikit-learn import failed" && exit 1)
	@echo ""
	@echo "6️⃣  Azure SDK:"
	@$(VENV_PYTHON) -c "from azure.storage.blob import BlobServiceClient; print('   Azure SDK ✅')" || \
		(echo "   ❌ Azure SDK import failed" && exit 1)
	@echo ""
	@echo "7️⃣  Streamlit:"
	@$(VENV_PYTHON) -c "import streamlit; print(f'   Streamlit {streamlit.__version__} ✅')" || \
		(echo "   ❌ Streamlit import failed" && exit 1)
	@echo ""
	@echo "8️⃣  Trained models:"
	@if [ -f "models/border_yolo.pt" ]; then \
		echo "   border_yolo.pt ✅"; \
	else \
		echo "   border_yolo.pt ⚠️  not found — pipeline will use yolov8n.pt fallback"; \
	fi
	@if [ -f "models/anomaly_model.pkl" ]; then \
		echo "   anomaly_model.pkl ✅"; \
	else \
		echo "   anomaly_model.pkl ⚠️  not found — anomaly detector will build baseline on first run"; \
	fi
	@echo ""
	@echo "✅ Verification complete!"
	@echo ""


# ── RUN ─────────────────────────────────

run:
	@echo "🚀 Running pipeline on test video..."
	@if [ ! -f "$(TEST_VIDEO)" ]; then \
		echo "⚠️  Test video not found at $(TEST_VIDEO)"; \
		echo "   Generating synthetic test video..."; \
		$(VENV_PYTHON) scripts/generate_test_video.py; \
	fi
	$(VENV_PYTHON) src/pipeline.py --video $(TEST_VIDEO) --save-frames

run-camera:
	@echo "🎥 Running pipeline on live camera (index 0)..."
	$(VENV_PYTHON) src/pipeline.py --camera 0

dashboard:
	@echo "📊 Launching Streamlit dashboard..."
	@echo "   Open your browser at: http://localhost:8501"
	@echo ""
	$(VENV_PYTHON) -m streamlit run dashboard/app.py

pilot:
	@echo "🔧 Running manual integration checker..."
	$(VENV_PYTHON) scripts/pilot.py $(TEST_VIDEO)

smoke:
	@echo "💨 Running smoke test..."
	$(VENV_PYTHON) scripts/smoke_test.py


# ── TESTING ─────────────────────────────

test:
	@echo "🧪 Running all tests with coverage..."
	$(VENV_PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "✅ Tests complete!"

test-unit:
	@echo "🧪 Running unit tests..."
	$(VENV_PYTHON) -m pytest tests/ -v -m "not integration"

coverage:
	@echo "📊 Generating HTML coverage report..."
	$(VENV_PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✅ Report generated: htmlcov/index.html"

report:
	@mkdir -p reports
	@echo "📄 Generating HTML test report..."
	$(VENV_PYTHON) -m pytest tests/ --html=reports/test_report.html --self-contained-html
	@echo "✅ Report generated: reports/test_report.html"


# ── CODE QUALITY ────────────────────────

lint:
	@echo "🔍 Checking code style (flake8)..."
	$(VENV_PYTHON) -m flake8 src/ tests/ --max-line-length=100 --exclude=venv
	@echo "✅ Style check passed"

format:
	@echo "✨ Formatting code (black)..."
	$(VENV_PYTHON) -m black src/ tests/ --line-length=100
	@echo "✅ Code formatted"

type-check:
	@echo "🔍 Type checking (mypy)..."
	$(VENV_PYTHON) -m mypy src/ --ignore-missing-imports
	@echo "✅ Type check passed"


# ── CLEANUP ─────────────────────────────

clean:
	@echo "🧹 Removing virtual environment..."
	@rm -rf $(VENV) && echo "✅ Done" || echo "⚠️  Nothing to remove"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache htmlcov .coverage reports build dist *.egg-info 2>/dev/null || true
	@echo "✅ Deep clean complete"


# ── DIAGNOSTICS ─────────────────────────

doctor:
	@echo ""
	@echo "🏥 Running diagnostics..."
	@echo ""
	@echo "System:"; uname -a 2>/dev/null || echo "N/A"
	@echo ""
	@echo "Python (system):"; which $(PYTHON) && $(PYTHON) --version || echo "Not found"
	@echo ""
	@echo "Virtual environment:"
	@if [ -d "$(VENV)" ]; then \
		echo "  ✅ exists at ./$(VENV)"; \
		$(VENV_PYTHON) --version; \
		$(VENV_PYTHON) -m pip --version; \
	else \
		echo "  ❌ not found — run 'make setup'"; \
	fi
	@echo ""
	@echo "Model files:"
	@ls -lh models/ 2>/dev/null || echo "  models/ directory not found"
	@echo ""
	@echo "Data directories:"
	@ls data/ 2>/dev/null || echo "  data/ directory not found"
	@echo ""


# ── SHORTHAND ───────────────────────────

s: setup
i: install
t: test
c: clean
v: verify
l: lint
f: format
r: run
d: dashboard

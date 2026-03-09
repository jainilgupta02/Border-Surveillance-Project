.PHONY: setup install clean test

setup:
	@echo "🚀 Setting up project environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install packaging==23.2
	@echo "✅ Environment setup complete!"

install:
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && python -c "import torch, cv2, streamlit; from ultralytics import YOLO; print('Environment OK')"

clean:
	rm -rf venv
.PHONY: setup install test clean

setup:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	fi
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install packaging==23.2
	echo "Environment ready"

install:
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && python testenv.py

clean:
	rm -rf venv
PYTHON = python3
VENV = .venv
PIP = $(VENV)/bin/pip

requirements:
	@if [ ! -d $(VENV) ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Installing requirements..."
	@$(PIP) install --upgrade pip >/dev/null
	@$(PIP) install -r test_script/requirements.txt
	@echo "All dependencies installed in $(VENV)"

run:
	python test_script/test_script.py
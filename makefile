# Directories
RAW_DIR := notebooks_raw
NB_DIR := notebooks

# Default target
help:
	@echo "Available commands:"
	@echo "  make help                     - Show this help message"
	@echo "  make notebook                 - Convert all .py in $(RAW_DIR) to .ipynb in $(NB_DIR)"
	@echo "  make notebook_raw             - Convert all .ipynb in $(NB_DIR) to .py in $(RAW_DIR)"
	@echo "  make notebook_raw FILE=name   - Convert one notebook (e.g., make notebook_raw FILE=test.ipynb)"
	@echo "  make launch                   - Launch Jupyter Notebook"

notebook:
	@mkdir -p $(NB_DIR)
	@for f in $(RAW_DIR)/*.py; do \
		python3 -m nbconvert --to notebook $$f --output-dir=$(NB_DIR); \
	done

notebook_raw:
	@mkdir -p $(RAW_DIR)
ifdef FILE
	@python3 -m nbconvert --to script $(NB_DIR)/$(FILE) --output-dir=$(RAW_DIR)
else
	@for f in $(NB_DIR)/*.ipynb; do \
		python3 -m nbconvert --to script $$f --output-dir=$(RAW_DIR); \
	done
endif

# Launch Jupyter
launch:
	python3 -m notebook
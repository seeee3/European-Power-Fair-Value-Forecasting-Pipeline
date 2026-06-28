.PHONY: setup run clean test

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[notebook]"
	@echo "\nDone. Copy .env.example → .env and add your API keys, then: make run"

run:
	.venv/bin/python run_pipeline.py

clean:
	rm -f data/raw/*.parquet data/processed/*.parquet
	rm -f outputs/*.json outputs/*.csv outputs/figures/*.png

notebook:
	.venv/bin/jupyter notebook notebooks/

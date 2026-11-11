# Data-Science-Toolbox
Created to store data science tools 

# Each should pass now
python -c "import ds_toolbox; print(ds_toolbox.__version__)"
ruff check src/ tests/ 2>&1 | tail -3
mypy src 2>&1 | tail -5
pytest --no-cov -q 2>&1 | tail -10
lint-imports 2>&1 | tail -15
clean_py:
	find src -type d -name "__pycache__" -exec rm -rf {} +


lint:
	ruff format src/rawformer/
	ruff check src/rawformer/
	mypy src/
	pylint src/


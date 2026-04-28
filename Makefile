clean_py:
	find src -type d -name "__pycache__" -exec rm -rf {} +


lint:
	ruff format src/rawformer/ tests/
	ruff check src/rawformer/ tests/
	mypy src/

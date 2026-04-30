clean_py:
	find src -type d -name "__pycache__" -exec rm -rf {} +


lint:
	@echo "LINTING"
	ruff format src/rawformer/ tests/
	ruff check src/rawformer/ tests/

	@echo ""
	@echo "TYPE CHECKING"
	mypy src/

	@echo ""
	@echo "TESTING"
	pytest -s

# Setting up
uv sync --extra cpu
uv sync --extra cuda

pre-commit install
pre-commit run --all-files


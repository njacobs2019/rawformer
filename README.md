This implements transformer architectures from pure pytorch


RUNNING:
- turn off beartype
- turn off asserts at interpreter level


Notes on API
`references/` slow hand-rolled reference implementations



# Setting up
uv sync --extra cpu
uv sync --extra cuda

pre-commit install
pre-commit run --all-files

uv pip install -e .

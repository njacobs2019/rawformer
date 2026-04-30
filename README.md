# rawformer

Transformers from scratch.


This library implements transformer architectures from pure PyTorch. It's not production ready, more so for my personal use and understanding.

Supported architectures:
- ViT

Supported positional encodings:
- Learned
- RoPE (1D and 2D)

## Installation

```bash
pip install rawformer
```

This library uses runtime checks to validate itself and throw better error messages early. NOTE: Beartype is incompatible with torch.compile.
- Turn off runtime type checking with env var: `BEARTYPE=0`
- Turn off python interpreter's assert statements with env var: `PYTHONOPTIMIZE=1`

## Developer install
```
uv sync --extra cpu
uv sync --extra cuda

pre-commit install
pre-commit run --all-files

uv pip install -e .
```

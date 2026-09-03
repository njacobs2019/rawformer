# rawformer

Transformers from scratch, in pure PyTorch. Not production-ready — this is for my own use and understanding.

**Architectures:** ViT
**Positional encodings:** Learned, AxialRoPE

## Installation

```bash
pip install rawformer
```

Two env vars control runtime checking:

- `BEARTYPE=1` — enable runtime type checking (default off)
- `PYTHONOPTIMIZE=1` — strip `assert` statements (default on)

Most errors will be caught by `ValueError`, it is pretty safe to strip asserts.

## Usage notes

### Exclude position parameters from weight decay

`ViT.no_weight_decay()` returns the parameter names that must not be decayed — the class token and everything in the position scheme:

```python
skip = model.no_weight_decay()
params = list(model.named_parameters())
optim = torch.optim.AdamW(
    [
        {"params": [p for n, p in params if n not in skip], "weight_decay": 0.05},
        {"params": [p for n, p in params if n in skip], "weight_decay": 0.0},
    ],
    lr=1e-3,
)
```

`AdamW(model.parameters(), weight_decay=...)` decays the learned position table and the class token, and the common `p.ndim >= 2` grouping rule doesn't save them — both are ndim-3.

### RoPE base is a buffer by default

`AxialRoPE`'s rotary base is a non-trainable buffer. Its gradient is structurally attenuated (the `i=0` channel contributes nothing, and the channels that do contribute are weighted by their own small θ), while decoupled weight decay acts on it regardless — so as a parameter it drifts toward 0, collapsing every rotary frequency to ~1.0 and destroying positional resolution over a long run. Pass `learnable=True` to opt in, and exclude it from decay via `no_weight_decay()`.

### Choosing `rotary_dim`

RoPE is applied per attention head, so `rotary_dim` must be `<= head_dim` (**not** `embed_dim`). Values below `head_dim` are valid and leave the remaining dimensions unrotated. Must be divisible by `2 * n_axes`.

### Axis order

`AxialRoPE` assigns `axes[k]` to `spatial_shape[k]`, and flattens the position grid row-major to match the tokenizer's output order. Per-axis `init_theta` must be given in the same order.

### Changing input resolution

`AxialRoPE` handles new input resolutions unchanged. `LearnedPositionEmbeddings` does not — it raises above `max_len`, with no position-embedding interpolation.

## Developer install

```bash
uv sync --extra cpu    # or --extra cuda
uv pip install -e .

pre-commit install
pre-commit run --all-files
```
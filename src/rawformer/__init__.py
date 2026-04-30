import os

if os.environ.get("BEARTYPE", "1") not in ("0", "false", "no"):
    print("BEARTYPE is ON")

    from jaxtyping import install_import_hook

    install_import_hook("rawformer", "beartype.beartype")

from .embedding import SimplePatchEmbedding
from .position_encoding import LearnedPositionEmbeddings, RoPE1D, RoPE2D, apply_rope

__all__ = [
    "LearnedPositionEmbeddings",
    "RoPE1D",
    "RoPE2D",
    "SimplePatchEmbedding",
    "apply_rope",
]

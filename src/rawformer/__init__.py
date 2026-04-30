import os

if os.environ.get("BEARTYPE", "1") not in ("0", "false", "no"):
    print("BEARTYPE is ON")

    from jaxtyping import install_import_hook

    install_import_hook("rawformer", "beartype.beartype")

from .rope import RoPE1D, RoPE2D
from .vit import LearnedPositionEmbeddings, SimplePatchEmbedding, ViT

__all__ = [
    "LearnedPositionEmbeddings",
    "RoPE1D",
    "RoPE2D",
    "SimplePatchEmbedding",
    "ViT",
]

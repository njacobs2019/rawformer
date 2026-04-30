import os

if os.environ.get("BEARTYPE", "1") not in ("0", "false", "no"):
    print("BEARTYPE is ON")

    from jaxtyping import install_import_hook

    install_import_hook("rawformer", "beartype.beartype")

from .embedding import SimplePatchEmbedding
from .position_encoding import LearnedPositionEmbeddings, RoPE2D

__all__ = [
    "LearnedPositionEmbeddings",
    "RoPE2D",
    "SimplePatchEmbedding",
]

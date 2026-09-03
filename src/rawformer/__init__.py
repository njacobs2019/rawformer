import os

if os.environ.get("BEARTYPE", "0").lower() in ("1", "true", "yes", "on"):
    print("BEARTYPE is ON")

    from jaxtyping import install_import_hook

    install_import_hook("rawformer", "beartype.beartype")

from .embedding import SimplePatchEmbedding
from .position_encoding import AxialRoPE, LearnedPositionEmbeddings
from .vit import ViT

__all__ = [
    "AxialRoPE",
    "LearnedPositionEmbeddings",
    "SimplePatchEmbedding",
    "ViT",
]

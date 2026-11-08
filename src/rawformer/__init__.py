import os

if os.environ.get("BEARTYPE", "0").lower() in ("1", "true", "yes", "on"):
    print("BEARTYPE is ON")

    try:
        import beartype  # noqa: F401
        from jaxtyping import install_import_hook
    except ImportError as e:
        msg = (
            "BEARTYPE=1 requires jaxtyping and beartype; "
            "install with `pip install rawformer[typecheck]`"
        )
        raise ImportError(msg) from e

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

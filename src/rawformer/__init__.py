import os

from .vit import LearnedPositionEmbeddings, SimplePatchEmbedding, ViT

__all__ = ["LearnedPositionEmbeddings", "SimplePatchEmbedding", "ViT"]


if os.environ.get("BEARTYPE", "1") not in ("0", "false", "no"):
    print("BEARTYPE is ON")
    from beartype.claw import beartype_this_package

    beartype_this_package()

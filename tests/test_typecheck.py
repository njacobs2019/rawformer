import os

import pytest
import torch

from rawformer.vit import ClassToken

pytestmark = pytest.mark.skipif(
    os.environ.get("BEARTYPE", "0").lower() not in ("1", "true", "yes", "on"),
    reason="runtime shape checking disabled",
)


def test_shape_checks_are_active() -> None:
    from jaxtyping import TypeCheckError

    cls_tok = ClassToken(embed_dim=4)
    bad = torch.randn(3, 4)  # 2D, but extract() expects Tokens = "b l d"

    # Plain torch would just return tokens[:, 0] here; only jaxtyping rejects it.
    with pytest.raises(TypeCheckError):
        cls_tok.extract(bad)

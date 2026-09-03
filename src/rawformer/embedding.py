"""
Handles tokenization and embedding
"""

import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class SimplePatchEmbedding(nn.Module):
    """
    Creates non-overlapping patches and linearly embeds them
    """

    grid_rank = 2  # forward() returns a 2D (h, w) grid

    def __init__(self, patch_size: int, channels: int, embed_dim: int) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.fc = nn.Linear(patch_size * patch_size * channels, embed_dim)

    def forward(
        self, x: Float[Tensor, "b c h w"]
    ) -> tuple[Float[Tensor, "b len embed_dim"], tuple[int, int]]:
        _b, _c, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            msg = (
                f"image size ({h}, {w}) must be divisible by patch_size "
                f"{self.patch_size}; otherwise the bottom/right edge is silently "
                f"cropped"
            )
            raise ValueError(msg)
        grid_size = (h // self.patch_size, w // self.patch_size)

        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = rearrange(patches, "b dim len -> b len dim")

        return self.fc(patches), grid_size

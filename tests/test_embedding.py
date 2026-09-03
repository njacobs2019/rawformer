import pytest
import torch

from rawformer import SimplePatchEmbedding


def test_simple_patch_embeddings() -> None:
    batch = 2
    img_size = 224
    patch_size = 14
    channels = 1
    embed_dim = 4

    embed = SimplePatchEmbedding(
        patch_size=patch_size, channels=channels, embed_dim=embed_dim
    )
    x = torch.rand(batch, channels, img_size, img_size)
    x, grid = embed(x)

    assert x.shape == (batch, (img_size // patch_size) ** 2, embed_dim)
    assert grid == (16, 16)


### THE BELOW TESTS WERE WRITTEN BY CLAUDE ###


@pytest.mark.parametrize(("h", "w"), [(225, 224), (224, 225), (225, 225), (100, 100)])
def test_patch_embedding_rejects_indivisible_image_size(h: int, w: int) -> None:
    """Must raise rather than let F.unfold silently crop the bottom/right edge.

    This is a ValueError, not an assert, so it survives PYTHONOPTIMIZE=1 -- the
    configuration the README recommends for training runs.
    """
    embed = SimplePatchEmbedding(patch_size=14, channels=3, embed_dim=8)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        embed(torch.rand(1, 3, h, w))


def test_patch_order_is_row_major() -> None:
    """Patch i*W+j must be grid cell (i, j); this is what AxialRoPE assumes."""
    h_patches, w_patches, patch = 2, 3, 2
    embed = SimplePatchEmbedding(patch_size=patch, channels=1, embed_dim=1)

    # make the fc layer average its patch, then stamp each patch with its index
    with torch.no_grad():
        embed.fc.weight.fill_(1.0 / (patch * patch))
        embed.fc.bias.fill_(0.0)

    img = torch.zeros(1, 1, h_patches * patch, w_patches * patch)
    for i in range(h_patches):
        for j in range(w_patches):
            img[0, 0, i * patch : (i + 1) * patch, j * patch : (j + 1) * patch] = (
                i * w_patches + j
            )

    tokens, grid = embed(img)
    assert grid == (h_patches, w_patches)
    expected = torch.arange(h_patches * w_patches, dtype=torch.float32)
    torch.testing.assert_close(tokens[0, :, 0], expected)


def test_patch_embedding_exposes_shape_metadata() -> None:
    """ViT.__init__ reads these to validate its components up front."""
    embed = SimplePatchEmbedding(patch_size=14, channels=3, embed_dim=64)
    assert embed.embed_dim == 64
    assert embed.grid_rank == 2

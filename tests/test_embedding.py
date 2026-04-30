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

import torch

from rawformer import LearnedPositionEmbeddings, RoPE2D
from rawformer.position_encoding import apply_rope


def test_learned_position_embeddings() -> None:
    batch = 2
    dim = 4
    length = 5

    embed = LearnedPositionEmbeddings(max_len=length, embed_dim=dim)
    x = torch.rand(batch, length, dim, dtype=torch.float32)

    x, cache = embed.prepare(x, (length,))

    assert x.shape == (batch, length, dim)
    assert cache is None


def test_2d_rope_embeddings() -> None:
    batch = 2
    h = 3
    w = 3
    dim = 8

    length = h * w
    rope = RoPE2D(rotary_dim=8)

    x = torch.ones(batch, length, dim)
    x, cache = rope.prepare(x, (h, w))
    y = apply_rope(x, cache)

    assert cache[0].shape == (length, dim)
    assert y.shape == x.shape

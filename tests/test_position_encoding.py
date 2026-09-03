import torch

from rawformer import AxialRoPE, LearnedPositionEmbeddings
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
    batch, h, w = 2, 3, 3
    num_heads, head_dim, rot_dim = 2, 9, 8
    length = h * w
    embed_dim = num_heads * head_dim

    rope = AxialRoPE(rotary_dim=rot_dim, n_axes=2)

    tokens = torch.ones(batch, length, embed_dim)
    tokens, cache = rope.prepare(tokens, (h, w))

    sin, _cos = cache
    assert sin.shape == (length, rot_dim)

    q = torch.ones(batch, num_heads, length, head_dim)
    q_rot = apply_rope(q, cache)
    assert q_rot.shape == q.shape


def test_3d_rope_embeddings() -> None:
    batch, c, h, w = 2, 3, 5, 7
    num_heads, head_dim, rot_dim = 2, 13, 12
    length = c * h * w
    embed_dim = num_heads * head_dim

    rope = AxialRoPE(rotary_dim=rot_dim, n_axes=3)

    tokens = torch.ones(batch, length, embed_dim)
    tokens, cache = rope.prepare(tokens, (c, h, w))

    sin, _cos = cache
    assert sin.shape == (length, rot_dim)

    q = torch.ones(batch, num_heads, length, head_dim)
    q_rot = apply_rope(q, cache)
    assert q_rot.shape == q.shape

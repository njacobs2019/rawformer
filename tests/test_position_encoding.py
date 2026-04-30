import torch

from rawformer import LearnedPositionEmbeddings


def test_learned_position_embeddings() -> None:
    batch = 2
    dim = 4
    length = 5

    embed = LearnedPositionEmbeddings(max_len=length, embed_dim=dim)
    x = torch.rand(batch, length, dim)
    x = embed(x)

    assert x.shape == (batch, length, dim)

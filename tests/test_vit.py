"""Basic unit tests for models"""

import torch
from torch import nn

from rawformer import LearnedPositionEmbeddings, SimplePatchEmbedding, ViT
from rawformer.vit import EncoderBlock


def test_simple_patch_embeddings() -> None:
    batch = 2
    img_size = 224
    patch_size = 14
    channels = 1
    embed_dim = 4

    embed = SimplePatchEmbedding(
        patch_len=patch_size, channels=channels, embed_dim=embed_dim
    )
    x = torch.rand(batch, channels, img_size, img_size)
    x = embed(x)

    assert x.shape == (batch, (img_size // patch_size) ** 2, embed_dim)


def test_learned_position_embeddings() -> None:
    batch = 2
    dim = 4
    length = 5

    embed = LearnedPositionEmbeddings(max_len=length, embed_dim=dim)
    x = torch.rand(batch, length, dim)
    x = embed(x)

    assert x.shape == (batch, length, dim)


def test_encoder_block_mha() -> None:
    batch = 2
    length = 5
    dim = 12
    num_heads = 3
    head_dim = dim // num_heads

    enc = EncoderBlock(
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_bias=False,
        mlp_hidden_dim=6,
        dropout=0.0,
        attn_dropout=0.0,
        attn_mask=None,
    )

    x = torch.rand(batch, length, dim)
    out = enc._mha(x)  # noqa
    assert out.shape == (batch, length, dim)


def test_encoder_block() -> None:
    batch = 2
    length = 5
    dim = 12
    num_heads = 3
    head_dim = dim // num_heads

    enc = EncoderBlock(
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_bias=False,
        mlp_hidden_dim=6,
        dropout=0.0,
        attn_dropout=0.0,
        attn_mask=None,
    )

    x = torch.rand(batch, length, dim)
    out = enc(x)
    assert out.shape == (batch, length, dim)


def test_vit_cls() -> None:
    # Test params
    batch = 2
    channels = 1
    dim = 12
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    patch_size = 14
    max_length = (img_size // patch_size) ** 2 + 1

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_len=patch_size, channels=channels, embed_dim=dim
    )

    pos_emb = LearnedPositionEmbeddings(max_len=max_length, embed_dim=dim)

    head = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    vit = ViT(
        num_layers=2,
        num_heads=num_heads,
        embed_dim=dim,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=True,
        patch_embedding=patch_emb,
        rope=None,
        position_embedding=pos_emb,
        use_cls_tok=True,
        head=head,
    )

    # Forward pass and check shape
    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, 1)


def test_vit_ae() -> None:
    # Test params
    batch = 2
    channels = 1
    dim = 12
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    patch_size = 14
    max_length = (img_size // patch_size) ** 2

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_len=patch_size, channels=channels, embed_dim=dim
    )

    pos_emb = LearnedPositionEmbeddings(max_len=max_length, embed_dim=dim)

    head = nn.Sequential(nn.Linear(dim, 3), nn.Sigmoid())

    vit = ViT(
        num_layers=2,
        num_heads=num_heads,
        embed_dim=dim,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=True,
        patch_embedding=patch_emb,
        rope=None,
        position_embedding=pos_emb,
        use_cls_tok=False,
        head=head,
    )

    # Forward pass and check shape
    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, max_length, 3)

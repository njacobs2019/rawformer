"""Basic unit tests for models"""

import torch
from torch import nn

from rawformer import LearnedPositionEmbeddings, RoPE2D, SimplePatchEmbedding
from rawformer.vit import EncoderBlock, ViT


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
    out = enc._mha(x, rope_cache=None)  # noqa
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
    out = enc(x, rope_cache=None)
    assert out.shape == (batch, length, dim)


def test_vit_dense() -> None:
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
        patch_size=patch_size, channels=channels, embed_dim=dim
    )
    pos_emb = LearnedPositionEmbeddings(max_len=max_length, embed_dim=dim)
    head = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        embed_dim=dim,
        mlp_hidden_dim=mlp_hidden_dim,
        use_cls=False,
    )

    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, max_length, 1)


def test_vit_dense_rope2d() -> None:
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
        patch_size=patch_size, channels=channels, embed_dim=dim
    )

    pos_emb = RoPE2D(rotary_dim=dim)
    head = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        embed_dim=dim,
        mlp_hidden_dim=mlp_hidden_dim,
        use_cls=False,
    )

    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, max_length, 1)


def test_vit_classifier() -> None:
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
        patch_size=patch_size, channels=channels, embed_dim=dim
    )
    pos_emb = LearnedPositionEmbeddings(max_len=max_length, embed_dim=dim)
    head = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        embed_dim=dim,
        mlp_hidden_dim=mlp_hidden_dim,
        use_cls=True,
    )

    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, 1)


def test_vit_classifier_rope2d() -> None:
    # Test params
    batch = 2
    channels = 1
    dim = 12
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    patch_size = 14

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_size=patch_size, channels=channels, embed_dim=dim
    )

    pos_emb = RoPE2D(rotary_dim=dim)
    head = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        embed_dim=dim,
        mlp_hidden_dim=mlp_hidden_dim,
        use_cls=True,
    )

    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, 1)

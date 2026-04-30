"""Basic unit tests for models"""

import pytest
import torch
from torch import nn

from rawformer import LearnedPositionEmbeddings, SimplePatchEmbedding
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


@pytest.mark.skip(reason="Waiting for VIT FIX")
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
        patch_size=patch_size, channels=channels, embed_dim=dim
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


@pytest.mark.skip(reason="Waiting for VIT FIX")
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
        patch_size=patch_size, channels=channels, embed_dim=dim
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

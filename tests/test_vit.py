"""Basic unit tests for models"""

import pytest
import torch
from torch import nn

from rawformer import AxialRoPE, LearnedPositionEmbeddings, SimplePatchEmbedding, ViT
from rawformer.vit import ClassToken, EncoderBlock


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
    )

    x = torch.rand(batch, length, dim)
    out = enc(x, rope_cache=None)
    assert out.shape == (batch, length, dim)


def test_vit_dense() -> None:
    # Test params
    batch = 2
    channels = 1
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    head_dim = 4
    patch_size = 14

    max_length = (img_size // patch_size) ** 2
    embed_dim = num_heads * head_dim

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_size=patch_size, channels=channels, embed_dim=embed_dim
    )
    pos_emb = LearnedPositionEmbeddings(max_len=max_length, embed_dim=embed_dim)
    head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        head_dim=head_dim,
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
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    head_dim = 4
    patch_size = 14

    max_length = (img_size // patch_size) ** 2
    embed_dim = num_heads * head_dim

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_size=patch_size, channels=channels, embed_dim=embed_dim
    )

    pos_emb = AxialRoPE(rotary_dim=head_dim, n_axes=2)
    head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        head_dim=head_dim,
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
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    head_dim = 4
    patch_size = 14

    max_length = (img_size // patch_size) ** 2
    embed_dim = num_heads * head_dim

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_size=patch_size, channels=channels, embed_dim=embed_dim
    )
    pos_emb = LearnedPositionEmbeddings(max_len=max_length, embed_dim=embed_dim)
    head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        head_dim=head_dim,
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
    img_size = 224
    mlp_hidden_dim = 5
    num_heads = 3
    head_dim = 4
    patch_size = 14

    embed_dim = num_heads * head_dim

    # Create objects
    patch_emb = SimplePatchEmbedding(
        patch_size=patch_size, channels=channels, embed_dim=embed_dim
    )

    pos_emb = AxialRoPE(rotary_dim=head_dim, n_axes=2)
    head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    vit = ViT(
        patch_emb,
        pos_emb,
        head,
        num_layers=2,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        use_cls=True,
    )

    x = torch.rand(batch, channels, img_size, img_size)
    y = vit(x)
    assert y.shape == (batch, 1)


### THE BELOW TESTS WERE WRITTEN BY CLAUDE ###


def make_vit(*, use_cls: bool = True, head_dim: int = 8, rotary_dim: int = 8) -> ViT:
    return ViT(
        SimplePatchEmbedding(patch_size=16, channels=3, embed_dim=4 * head_dim),
        AxialRoPE(rotary_dim=rotary_dim, n_axes=2),
        None,
        num_layers=2,
        num_heads=4,
        head_dim=head_dim,
        mlp_hidden_dim=32,
        use_cls=use_cls,
    )


def test_cls_token_receives_identity_rotation() -> None:
    """The prepended cls token must not be rotated (sin=0, cos=1)."""
    rope = AxialRoPE(rotary_dim=8, n_axes=2)
    cache = rope.build_cache((3, 3), dtype=torch.float32)
    cls = ClassToken(embed_dim=8)

    tokens, new_cache = cls.prepend(torch.zeros(2, 9, 8), cache)
    assert new_cache is not None
    sin, cos = new_cache

    assert tokens.shape == (2, 10, 8)
    assert sin.shape == (10, 8)
    torch.testing.assert_close(sin[0], torch.zeros(8))
    torch.testing.assert_close(cos[0], torch.ones(8))
    # the patch rows must be shifted down by exactly one, not resampled
    torch.testing.assert_close(sin[1:], cache[0])
    torch.testing.assert_close(cos[1:], cache[1])


def test_cls_token_prepend_is_noop_without_rope() -> None:
    cls = ClassToken(embed_dim=8)
    tokens, cache = cls.prepend(torch.zeros(2, 9, 8), None)
    assert cache is None
    assert tokens.shape == (2, 10, 8)


def test_vit_rejects_patch_embed_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="patch_embed produces"):
        ViT(
            SimplePatchEmbedding(patch_size=16, channels=3, embed_dim=99),
            AxialRoPE(rotary_dim=8, n_axes=2),
            None,
            num_layers=1,
            num_heads=4,
            head_dim=8,
            mlp_hidden_dim=32,
            use_cls=True,
        )


def test_vit_rejects_rotary_dim_larger_than_head_dim() -> None:
    """Guards the f02cf19 API change: rotary_dim is sized by head_dim, not embed_dim."""
    with pytest.raises(ValueError, match="rotary_dim"):
        make_vit(head_dim=8, rotary_dim=32)  # 32 == embed_dim, the stale usage


def test_vit_rejects_axis_count_mismatch() -> None:
    with pytest.raises(ValueError, match="n_axes"):
        ViT(
            SimplePatchEmbedding(patch_size=16, channels=3, embed_dim=32),
            # 3D rope on a 2D patch grid; rotary_dim=6 is divisible by 2*3 and
            # <= head_dim, so this reaches the axis-count check rather than
            # tripping an earlier one
            AxialRoPE(rotary_dim=6, n_axes=3),
            None,
            num_layers=1,
            num_heads=4,
            head_dim=8,
            mlp_hidden_dim=32,
            use_cls=True,
        )


def test_vit_accepts_rotary_dim_below_head_dim() -> None:
    make_vit(head_dim=16, rotary_dim=8)  # partial rotation is legitimate


def test_no_weight_decay_covers_cls_token_and_position_params() -> None:
    vit = ViT(
        SimplePatchEmbedding(patch_size=16, channels=3, embed_dim=32),
        LearnedPositionEmbeddings(max_len=196, embed_dim=32),
        None,
        num_layers=1,
        num_heads=4,
        head_dim=8,
        mlp_hidden_dim=32,
        use_cls=True,
    )
    skip = vit.no_weight_decay()
    assert "cls_tok.tok" in skip
    assert "pos_embed.E" in skip

    names = {n for n, _ in vit.named_parameters()}
    assert skip <= names, "no_weight_decay must return real parameter names"
    # ordinary weights must still be decayed
    assert not any(n.startswith("layers.") for n in skip)


def test_learnable_rope_base_is_reported_by_no_weight_decay() -> None:
    vit = ViT(
        SimplePatchEmbedding(patch_size=16, channels=3, embed_dim=32),
        AxialRoPE(rotary_dim=8, n_axes=2, learnable=True),
        None,
        num_layers=1,
        num_heads=4,
        head_dim=8,
        mlp_hidden_dim=32,
        use_cls=True,
    )
    skip = vit.no_weight_decay()
    assert any("log_base" in n for n in skip)


def test_weight_decay_cannot_collapse_the_rope_base() -> None:
    """C1 regression: naive AdamW(weight_decay=..) must not erode the rotary base.

    A decayed log_base drives every rotary frequency toward 1.0, destroying
    positional resolution with no error and no NaN -- only a worse loss curve.
    """
    torch.manual_seed(0)
    vit = make_vit()
    before = vit.pos_embed.axes[0].log_base.clone()

    # deliberately the naive setup that ignores no_weight_decay()
    optim = torch.optim.AdamW(vit.parameters(), lr=1e-2, weight_decay=0.5)
    for _ in range(50):
        optim.zero_grad()
        vit(torch.randn(2, 3, 64, 64)).square().mean().backward()
        optim.step()

    torch.testing.assert_close(vit.pos_embed.axes[0].log_base, before)

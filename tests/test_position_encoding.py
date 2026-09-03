import math

import pytest
import torch
from torch import Tensor

from rawformer import AxialRoPE, LearnedPositionEmbeddings
from rawformer.position_encoding import RoPE1D, apply_rope


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


### THE BELOW TESTS WERE WRITTEN BY CLAUDE ###


def reference_axial_rope(
    x: Tensor,
    spatial_shape: tuple[int, ...],
    rotary_dim: int,
    base: float = 10_000.0,
) -> Tensor:
    """Independent closed-form axial RoPE, written from the RoFormer definition.

    Deliberately shares no code with the implementation: it rotates each 2D pair
    with explicit sin/cos rather than the stacked-flip trick. This pins the sign
    convention, the pairing convention (interleaved, not half-split) and the
    frequency schedule all at once. `x` is (b, heads, len, head_dim).
    """
    n_axes = len(spatial_shape)
    axis_dim = rotary_dim // n_axes
    half = axis_dim // 2

    # position of every token along every axis, in row-major order
    coords = torch.cartesian_prod(*[torch.arange(e) for e in spatial_shape])
    coords = coords.reshape(-1, n_axes)

    out = x.clone()
    for k in range(n_axes):
        for j in range(half):
            theta = base ** (-2.0 * j / axis_dim)
            angle = coords[:, k].to(x.dtype) * theta  # (len,)
            cos, sin = torch.cos(angle), torch.sin(angle)

            lo = k * axis_dim + 2 * j  # even slot of this pair
            hi = lo + 1  # odd slot of this pair
            a, b = x[..., lo], x[..., hi]
            out[..., lo] = a * cos - b * sin
            out[..., hi] = b * cos + a * sin
    return out


# --------------------------------------------------------------------------
# Value-level tests. The shape-only tests above cannot distinguish a correct
# rotation from a sign-flipped one or from the llama/NeoX half-split
# convention -- both preserve shape exactly. These pin the actual numbers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spatial_shape", "rotary_dim"),
    [
        ((3, 3), 8),  # square grid
        ((4, 7), 8),  # rectangular: catches a transposed axis broadcast
        ((7, 4), 12),
        ((2, 3, 5), 12),  # 3 axes
    ],
)
def test_apply_rope_matches_closed_form(
    spatial_shape: tuple[int, ...], rotary_dim: int
) -> None:
    """The anchor test: rotation must match an independent closed-form reference."""
    torch.manual_seed(0)
    n_axes = len(spatial_shape)
    length = math.prod(spatial_shape)
    head_dim = rotary_dim + 3  # leave a non-rotary tail

    rope = AxialRoPE(rotary_dim=rotary_dim, n_axes=n_axes)
    cache = rope.build_cache(spatial_shape, dtype=torch.float32)

    q = torch.randn(2, 3, length, head_dim)
    got = apply_rope(q, cache)
    want = reference_axial_rope(q, spatial_shape, rotary_dim)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_apply_rope_is_identity_at_position_zero() -> None:
    rope = AxialRoPE(rotary_dim=8, n_axes=2)
    cache = rope.build_cache((3, 3), dtype=torch.float32)
    q = torch.randn(2, 2, 9, 8)

    # token 0 sits at grid coord (0, 0) -> zero rotation on every axis
    torch.testing.assert_close(apply_rope(q, cache)[:, :, 0], q[:, :, 0])


def test_apply_rope_leaves_non_rotary_tail_untouched() -> None:
    rot_dim, head_dim = 8, 20
    rope = AxialRoPE(rotary_dim=rot_dim, n_axes=2)
    cache = rope.build_cache((3, 3), dtype=torch.float32)

    q = torch.randn(2, 2, 9, head_dim)
    out = apply_rope(q, cache)

    assert torch.equal(out[..., rot_dim:], q[..., rot_dim:])
    assert not torch.allclose(out[..., :rot_dim], q[..., :rot_dim])


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    """The defining RoPE property, in 2D."""
    torch.manual_seed(0)
    h, w, rot_dim = 6, 6, 8
    rope = AxialRoPE(rotary_dim=rot_dim, n_axes=2)
    cache = rope.build_cache((h, w), dtype=torch.float32)

    # identical content at every position, so only position can affect the score
    q = torch.randn(rot_dim).expand(1, 1, h * w, rot_dim).contiguous()
    k = torch.randn(rot_dim).expand(1, 1, h * w, rot_dim).contiguous()
    scores = apply_rope(q, cache)[0, 0] @ apply_rope(k, cache)[0, 0].T

    dy, dx = 1, 2
    ref = scores[0 * w + 0, dy * w + dx]
    for i in range(h - dy):
        for j in range(w - dx):
            got = scores[i * w + j, (i + dy) * w + (j + dx)]
            torch.testing.assert_close(got, ref, rtol=0, atol=1e-5)


def test_rope_axes_are_not_interchangeable() -> None:
    """Offset (1,0) must score differently from (0,1); catches collapsed axes."""
    torch.manual_seed(0)
    h, w, rot_dim = 5, 5, 8
    rope = AxialRoPE(rotary_dim=rot_dim, n_axes=2)
    # give the two axes different bases so they are genuinely distinguishable
    with torch.no_grad():
        rope.axes[1].log_base.fill_(math.log(100.0))
    cache = rope.build_cache((h, w), dtype=torch.float32)

    q = torch.randn(rot_dim).expand(1, 1, h * w, rot_dim).contiguous()
    k = torch.randn(rot_dim).expand(1, 1, h * w, rot_dim).contiguous()
    scores = apply_rope(q, cache)[0, 0] @ apply_rope(k, cache)[0, 0].T

    down = scores[0 * w + 0, 1 * w + 0]  # offset (1, 0)
    right = scores[0 * w + 0, 0 * w + 1]  # offset (0, 1)
    assert not torch.isclose(down, right, atol=1e-4)


def test_rope_grid_order_is_row_major() -> None:
    """Cache row i*W+j must encode grid coord (i, j), matching F.unfold order."""
    h, w, rot_dim = 3, 4, 8
    rope = AxialRoPE(rotary_dim=rot_dim, n_axes=2)
    sin, cos = rope.build_cache((h, w), dtype=torch.float32)
    axis_dim = rot_dim // 2

    per_axis = RoPE1D(axis_dim)
    sin_r, cos_r = per_axis.build_cache(max(h, w), dtype=torch.float32)

    for i in range(h):
        for j in range(w):
            row = i * w + j
            torch.testing.assert_close(sin[row, :axis_dim], sin_r[i])
            torch.testing.assert_close(cos[row, :axis_dim], cos_r[i])
            torch.testing.assert_close(sin[row, axis_dim:], sin_r[j])
            torch.testing.assert_close(cos[row, axis_dim:], cos_r[j])


def test_rope_base_is_buffer_by_default_and_learnable_on_request() -> None:
    default = AxialRoPE(rotary_dim=8, n_axes=2)
    assert list(default.parameters()) == [], (
        "log_base must not be a Parameter by default: it gets ~no gradient signal "
        "but full weight decay, which collapses every rotary frequency to ~1.0"
    )
    assert "axes.0.log_base" in default.state_dict()

    learnable = AxialRoPE(rotary_dim=8, n_axes=2, learnable=True)
    assert len(list(learnable.parameters())) == 2

    # both must produce identical caches at init
    torch.testing.assert_close(
        default.build_cache((3, 3), dtype=torch.float32)[0],
        learnable.build_cache((3, 3), dtype=torch.float32)[0],
    )


def test_rope_cache_follows_module_dtype_and_device() -> None:
    rope = AxialRoPE(rotary_dim=8, n_axes=2).to(torch.float64)
    assert rope.axes[0].log_base.dtype == torch.float64
    sin, _cos = rope.build_cache((3, 3), dtype=torch.float32)
    assert sin.dtype == torch.float32


def test_axial_rope_rejects_indivisible_rotary_dim() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AxialRoPE(rotary_dim=6, n_axes=4)


def test_axial_rope_rejects_wrong_grid_rank() -> None:
    rope = AxialRoPE(rotary_dim=8, n_axes=2)
    with pytest.raises(ValueError, match="n_axes"):
        rope.build_cache((3, 3, 3), dtype=torch.float32)


def test_rope1d_rejects_odd_rotary_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        RoPE1D(rotary_dim=7)


def test_learned_position_embeddings_reject_overlong_sequence() -> None:
    embed = LearnedPositionEmbeddings(max_len=4, embed_dim=8)
    with pytest.raises(ValueError, match="exceeds max_len"):
        embed(torch.rand(2, 5, 8))

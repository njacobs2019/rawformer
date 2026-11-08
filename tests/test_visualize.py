# Rope visualization test

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange

from rawformer import AxialRoPE
from rawformer.position_encoding import apply_rope


@torch.no_grad()
def test_visualization() -> None:
    batch, num_heads = 1, 1
    dim = 24
    h = w = 32
    seq_len, d_head = h * w, dim

    thetas = [0.1, 1, 10, 100, 1000, 10_000, 100_000, 1_000_000]

    ncols = 4
    nrows = math.ceil(len(thetas) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()

    for ax, theta in zip(flat_axes, thetas, strict=False):
        rope = AxialRoPE(rotary_dim=dim, n_axes=2, init_theta=float(theta))
        cache = rope.build_cache((h, w), dtype=torch.float32)

        pre_rot = torch.ones(batch, num_heads, seq_len, d_head)
        pre_rot = F.normalize(pre_rot, dim=-1)

        # CHECK INPUT IS UNIT LENGTH (should be by construction)
        pre_norm = pre_rot.norm(dim=-1)
        assert torch.allclose(pre_norm, torch.ones_like(pre_norm), atol=1e-6), (
            f"pre_rot norm off: min={pre_norm.min():.6f} max={pre_norm.max():.6f}"
        )

        # APPLY ROPE
        after_rot = apply_rope(pre_rot, cache)

        # CHECK THAT ROPE PRESERVES NORM
        post_norm = after_rot.norm(dim=-1)
        assert torch.allclose(post_norm, torch.ones_like(post_norm), atol=1e-5), (
            f"apply_rope changed norm: min={post_norm.min():.6f}"
            f" max={post_norm.max():.6f}"
        )

        after_img = rearrange(after_rot, "b heads (h w) d -> b heads d h w", h=h, w=w)

        ref = after_img[:, :, :, h // 2 : h // 2 + 1, w // 2 : w // 2 + 1]
        sim = (after_img * ref).sum(dim=2).squeeze()

        im = ax.imshow(sim.detach().cpu().numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"θ = {theta:g}")
        ax.set_xticks([])
        ax.set_yticks([])

        ## CHECK TRANSLATIONAL INVARIANCE
        q_vec = F.normalize(torch.randn(batch, num_heads, 1, d_head), dim=-1)
        k_vec = F.normalize(torch.randn(batch, num_heads, 1, d_head), dim=-1)

        q = q_vec.expand(batch, num_heads, seq_len, d_head).contiguous()
        k = k_vec.expand(batch, num_heads, seq_len, d_head).contiguous()

        qr, kr = apply_rope(q, cache), apply_rope(k, cache)

        qi = rearrange(qr, "b nh (h w) d -> b nh d h w", h=h, w=w)
        ki = rearrange(kr, "b nh (h w) d -> b nh d h w", h=h, w=w)

        # (0,0)·(1,1) should equal (5,7)·(6,8) — same relative offset
        a = (qi[..., 0, 0] * ki[..., 1, 1]).sum(-1)
        b = (qi[..., 5, 7] * ki[..., 6, 8]).sum(-1)
        assert torch.allclose(a, b, atol=1e-5), (
            f"not shift-equivariant: {a.item()} vs {b.item()}"
        )

    for ax in flat_axes[len(thetas) :]:
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig("rope_theta_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    test_visualization()

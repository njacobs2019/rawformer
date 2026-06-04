"""
Handles positional encoding
"""

from typing import Protocol, runtime_checkable

import torch
from einops import rearrange
from torch import nn

from .types import RoPECache, Tokens


@runtime_checkable
class PositionScheme(Protocol):
    def prepare(
        self,
        x: Tokens,
        spatial_shape: tuple[int, ...],
    ) -> tuple[Tokens, RoPECache | None]: ...


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, max_len: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.max_len = max_len
        self.E = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(1, max_len, embed_dim))
        )  # Same as ViT and BERT
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tokens) -> Tokens:
        length = x.shape[1]
        assert length <= self.max_len

        return self.dropout(x + self.E[:, :length, :])

    def prepare(
        self,
        x: Tokens,
        spatial_shape: tuple[int, ...],  # noqa: ARG002
    ) -> tuple[Tokens, None]:
        return self(x), None


class RoPE1D(nn.Module):
    """1D RoPE implementation"""

    def __init__(self, rotary_dim: int, init_theta: float = 10_000) -> None:
        super().__init__()

        assert rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim

        init_base_theta = torch.tensor(init_theta, dtype=torch.float32)
        self.log_base = nn.Parameter(init_base_theta.log())

    def build_cache(self, length: int, dtype: torch.dtype | None = None) -> RoPECache:
        # Builds the `rotation matrix`

        half_dim = self.rotary_dim // 2
        device = self.log_base.device

        i = torch.arange(half_dim, device=device, dtype=dtype)
        theta = torch.exp(-i / half_dim * self.log_base)
        positions = torch.arange(length, device=device, dtype=dtype)
        freqs = torch.outer(positions, theta)  # (length, half_dim)

        sin = freqs.sin().repeat_interleave(2, dim=-1)  # (length, half_dim)
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # (length, half_dim)

        if dtype is not None:
            sin = sin.to(dtype)
            cos = cos.to(dtype)

        return sin, cos


class RoPE2D(nn.Module):
    """
    2D rope implementation where top is y and bottom half is x
      - X and Y have equal rotary dims
      - X and Y have separate learnable bases
    """

    def __init__(self, rotary_dim: int, init_theta: float = 10_000.0) -> None:
        super().__init__()

        assert rotary_dim % 4 == 0
        axis_dim = rotary_dim // 2
        self.rope_y = RoPE1D(axis_dim, init_theta)
        self.rope_x = RoPE1D(axis_dim, init_theta)

    def build_cache(
        self, h: int, w: int, dtype: torch.dtype | None = None
    ) -> RoPECache:

        sin_y, cos_y = self.rope_y.build_cache(h, dtype)  # (h, axis_dim)
        sin_x, cos_x = self.rope_x.build_cache(w, dtype)  # (w, axis_dim)

        sin_y = sin_y[:, None, :].expand(h, w, -1)
        cos_y = cos_y[:, None, :].expand(h, w, -1)
        sin_x = sin_x[None, :, :].expand(h, w, -1)
        cos_x = cos_x[None, :, :].expand(h, w, -1)

        # combine y|x
        sin = torch.cat((sin_y, sin_x), dim=-1)  # (h, w, rotary_dim)
        cos = torch.cat((cos_y, cos_x), dim=-1)  # (h, w, rotary_dim)

        sin = sin.reshape(h * w, -1)
        cos = cos.reshape(h * w, -1)

        return sin, cos

    def prepare(
        self, x: Tokens, spatial_shape: tuple[int, int]
    ) -> tuple[Tokens, RoPECache]:
        return x, self.build_cache(*spatial_shape, dtype=x.dtype)


class RoPE3D(nn.Module):
    """
    3D rope implementation
    Ordered y (top third) | x (middle third) | channel (bottom third)
    """

    def __init__(self, rotary_dim: int, init_theta: float = 10_000.0) -> None:
        super().__init__()

        assert rotary_dim % 6 == 0
        axis_dim = rotary_dim // 3
        self.rope_y = RoPE1D(axis_dim, init_theta)
        self.rope_x = RoPE1D(axis_dim, init_theta)
        self.rope_c = RoPE1D(axis_dim, init_theta)

    def build_cache(
        self, c: int, h: int, w: int, dtype: torch.dtype | None = None
    ) -> RoPECache:
        sin_y, cos_y = self.rope_y.build_cache(h, dtype)  # (h, axis_dim)
        sin_x, cos_x = self.rope_x.build_cache(w, dtype)  # (w, axis_dim)
        sin_c, cos_c = self.rope_c.build_cache(c, dtype)  # (c, axis_dim)

        # Expand dims
        sin_y = sin_y[None, :, None, :].expand(c, h, w, -1)
        cos_y = cos_y[None, :, None, :].expand(c, h, w, -1)
        sin_x = sin_x[None, None, :, :].expand(c, h, w, -1)
        cos_x = cos_x[None, None, :, :].expand(c, h, w, -1)
        sin_c = sin_c[:, None, None, :].expand(c, h, w, -1)
        cos_c = cos_c[:, None, None, :].expand(c, h, w, -1)

        # combine y|x|c
        sin = torch.cat((sin_y, sin_x, sin_c), dim=-1)  # (c, h, w, rotary_dim)
        cos = torch.cat((cos_y, cos_x, cos_c), dim=-1)  # (c, h, w, rotary_dim)

        sin = rearrange(sin, "c h w dim -> (c h w) dim")
        cos = rearrange(cos, "c h w dim -> (c h w) dim")

        return sin, cos

    def prepare(
        self, x: Tokens, spatial_shape: tuple[int, int, int]
    ) -> tuple[Tokens, RoPECache]:
        return x, self.build_cache(*spatial_shape, x.dtype)


def apply_rope(
    x: Tokens,
    rope_cache: RoPECache,
) -> Tokens:
    # Uses efficient 2D rotation implementation from RoFormer paper eqn 34

    # Unpack input and validate
    sin_mat, cos_mat = rope_cache
    rot_dim = sin_mat.shape[-1]
    assert rot_dim <= x.shape[-1], "rot_dim must be <= full dim"

    # Split x by rot_dim (only rotate top)
    x_top = x[..., :rot_dim]
    x_bottom = x[..., rot_dim:]

    # Reorder into [-x_2, x_1, -x_4, x_3, ...]
    x_even = x_top[..., ::2]
    x_odd = x_top[..., 1::2]
    x_top_flipped = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x_top)

    # Compute rotations
    x_rot = (x_top * cos_mat) + (x_top_flipped * sin_mat)

    return torch.cat((x_rot, x_bottom), dim=-1)

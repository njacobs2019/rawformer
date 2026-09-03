"""
Handles positional encoding
"""

from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor, nn

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
        if length > self.max_len:
            msg = f"sequence length {length} exceeds max_len {self.max_len}. "
            raise ValueError(msg)

        return self.dropout(x + self.E[:, :length, :])

    def prepare(
        self,
        x: Tokens,
        spatial_shape: tuple[int, ...],  # noqa: ARG002
    ) -> tuple[Tokens, None]:
        return self(x), None


class RoPE1D(nn.Module):
    """1D RoPE implementation

    The rotary base is stored in log space. It is a **buffer by default**: the base
    receives almost no gradient signal, but as a parameter it is fully exposed to
    weight decay, which drives ``log_base`` toward 0 and collapses every rotary
    frequency to ~1.0 -- silently destroying positional resolution over a long run.
    Pass ``learnable=True`` to opt in, and exclude it from weight decay if you do
    (see :meth:`rawformer.ViT.no_weight_decay`).
    """

    log_base: Tensor

    def __init__(
        self,
        rotary_dim: int,
        init_theta: float = 10_000,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        if rotary_dim % 2 != 0:
            msg = f"rotary_dim must be even, got {rotary_dim}"
            raise ValueError(msg)
        self.rotary_dim = rotary_dim

        log_theta = torch.tensor(init_theta, dtype=torch.float32).log()
        if learnable:
            self.log_base = nn.Parameter(log_theta)
        else:
            self.register_buffer("log_base", log_theta)

    def build_cache(self, length: int, dtype: torch.dtype) -> RoPECache:
        # Builds the `rotation matrix`

        half_dim = self.rotary_dim // 2
        device = self.log_base.device

        with torch.autocast(device_type=device.type, enabled=False):
            i = torch.arange(half_dim, device=device, dtype=torch.float32)
            theta = torch.exp(-i / half_dim * self.log_base)
            positions = torch.arange(length, device=device, dtype=torch.float32)
            freqs = torch.outer(positions, theta)  # (length, half_dim)

            sin = freqs.sin().repeat_interleave(2, dim=-1)  # (length, rotary_dim)
            cos = freqs.cos().repeat_interleave(2, dim=-1)  # (length, rotary_dim)

        return sin.to(dtype), cos.to(dtype)


class AxialRoPE(nn.Module):
    """
    N-dimensional axial RoPE. Splits rotary_dim evenly across axes,
    each with its own learnable base. Axis order matches spatial_shape.
    """

    def __init__(
        self,
        rotary_dim: int,
        n_axes: int,
        init_theta: float = 10_000.0,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if rotary_dim % (2 * n_axes) != 0:
            msg = (
                f"rotary_dim {rotary_dim} must be divisible by {2 * n_axes} "
                f"so each of the {n_axes} axes gets an even number of dimensions"
            )
            raise ValueError(msg)
        self.rotary_dim = rotary_dim
        self.n_axes = n_axes

        self.axes: list[RoPE1D] = [
            RoPE1D(rotary_dim // n_axes, init_theta, learnable=learnable)
            for _ in range(n_axes)
        ]
        self._axes_module_list = nn.ModuleList(self.axes)  # registration only

    def build_cache(
        self, spatial_shape: tuple[int, ...], dtype: torch.dtype
    ) -> RoPECache:
        if len(spatial_shape) != self.n_axes:
            msg = (
                f"spatial_shape {spatial_shape} has {len(spatial_shape)} axes but "
                f"this AxialRoPE was built for n_axes={self.n_axes}"
            )
            raise ValueError(msg)

        sins: list[Tensor] = []
        coss: list[Tensor] = []
        for k, (rope, extent) in enumerate(zip(self.axes, spatial_shape, strict=True)):
            sin_k, cos_k = rope.build_cache(extent, dtype=dtype)  # (extent, axis_dim)

            # broadcast axis k across the full grid
            view = [1] * self.n_axes + [-1]
            view[k] = extent
            target = (*spatial_shape, -1)
            sins.append(sin_k.view(view).expand(target))
            coss.append(cos_k.view(view).expand(target))

        sin = torch.cat(sins, dim=-1).reshape(-1, self.rotary_dim)
        cos = torch.cat(coss, dim=-1).reshape(-1, self.rotary_dim)

        return sin, cos

    def prepare(
        self, x: Tokens, spatial_shape: tuple[int, ...]
    ) -> tuple[Tokens, RoPECache]:
        return x, self.build_cache(spatial_shape, dtype=x.dtype)


def apply_rope(
    x: Float[Tensor, "b num_heads l d_head"],
    rope_cache: RoPECache,
) -> Float[Tensor, "b num_heads l d_head"]:
    # Uses efficient 2D rotation implementation from RoFormer paper eqn 34

    # Unpack input and validate
    sin_mat, cos_mat = rope_cache
    rot_dim = sin_mat.shape[-1]
    assert rot_dim <= x.shape[-1], "rot_dim must be <= head_dim"
    assert sin_mat.shape[-2] == x.shape[-2], "sequence length must match"

    # Split x by rot_dim (only rotate top)
    x_top = x[..., :rot_dim]  # (b, heads, l, rot_dim)
    x_bottom = x[..., rot_dim:]

    # Reorder into [-x_2, x_1, -x_4, x_3, ...]
    x_even = x_top[..., ::2]
    x_odd = x_top[..., 1::2]
    x_top_flipped = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x_top)

    # Compute rotations
    x_rot = (x_top * cos_mat) + (x_top_flipped * sin_mat)

    return torch.cat((x_rot, x_bottom), dim=-1)

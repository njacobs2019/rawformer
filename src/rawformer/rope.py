import torch
from jaxtyping import Float
from torch import Tensor, nn


class RoPE1D(nn.Module):
    """1D RoPE implementation"""

    def __init__(self, rotary_dim: int, init_theta: float = 10_000) -> None:
        super().__init__()

        assert rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim

        init_base_theta = torch.tensor(init_theta, dtype=torch.float32)
        self.log_base = nn.Parameter(init_base_theta.log())

    def build_cache(
        self, length: int
    ) -> tuple[Float[Tensor, "len rot_dim"], Float[Tensor, "len rot_dim"]]:
        # Builds the `rotation matrix`

        half_dim = self.rotary_dim // 2
        device, dtype = self.log_base.device, self.log_base.dtype

        i = torch.arange(half_dim, device=device, dtype=dtype)
        theta = torch.exp(-i / half_dim * self.log_base)
        positions = torch.arange(length, device=device, dtype=dtype)
        freqs = torch.outer(positions, theta)  # (length, half_dim)

        sin = freqs.sin().repeat_interleave(2, dim=-1)  # (length, half_dim)
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # (length, half_dim)

        return sin, cos


def apply_rope(
    x: Float[Tensor, "b len dim"],
    rope_cache: tuple[Float[Tensor, "len rot_dim"], Float[Tensor, "len rot_dim"]],
) -> Float[Tensor, "b len dim"]:
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

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
        self.log_base_theta = nn.Parameter(init_base_theta.log())

    def build_rot_mat(
        self, length: int
    ) -> tuple[Float[Tensor, "len rot_dim"], Float[Tensor, "len rot_dim"]]:
        # Builds the `rotation matrix`

        base_theta = self.log_base_theta.exp()
        num_rotary_freqs = self.rotary_dim // 2

        i = torch.arange(
            0, num_rotary_freqs, device=base_theta.device, dtype=base_theta.dtype
        )
        theta = base_theta ** (-i / num_rotary_freqs)
        positions = torch.arange(0, length)
        freqs = torch.einsum(
            "i,j -> ij", positions, theta
        )  # (length, num_rotary_freqs)

        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (length, rotary_dim)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (length, rotary_dim)

        return sin, cos


def apply_rope(
    x: Float[Tensor, "b len dim"],
    matrices: tuple[Float[Tensor, "len rot_dim"], Float[Tensor, "len rot_dim"]],
) -> Float[Tensor, "b len dim"]:
    # Uses efficient 2D rotation implementation from RoFormer paper eqn 34

    # Unpack input and validate
    sin_mat, cos_mat = matrices
    rot_dim = sin_mat.shape[-1]
    assert rot_dim <= x.shape[-1], "rot_dim must be <= full dim"

    # Split x by rot_dim
    x_top = x[..., :rot_dim]
    x_bottom = x[..., rot_dim:]

    # Create flipped vector
    x1 = x_top[..., ::2]
    x2 = x_top[..., 1::2]
    x_top_flipped = torch.stack((-x2, x1), dim=-1).reshape_as(x_top)

    # Compute rotations
    x_rot = (x_top * cos_mat) + (x_top_flipped * sin_mat)

    return torch.cat((x_rot, x_bottom), dim=-1)

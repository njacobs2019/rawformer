import torch
from torch import nn


def rotate_every_two(x):
    # x: (..., dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(x, sin, cos):
    # x: (..., dim)
    # sin, cos: broadcastable to x[..., :dim]
    x_rot = (x * cos) + (rotate_every_two(x) * sin)
    return x_rot


class LearnableFreqRoPE(nn.Module):
    def __init__(self, rotary_dim, max_pos, device=None):
        """
        rotary_dim: even number of rotary dims (per head)
        max_pos: maximum coordinate (like window size or image size)
        """
        super().__init__()
        assert rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim
        self.max_pos = max_pos
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize inv_freq with usual formula and then store log
        dim_idx = torch.arange(0, rotary_dim // 2, device=device).float()  # pairs
        init_inv_freq = 1.0 / (10000 ** (dim_idx / (rotary_dim // 2)))
        # store log so parameter is unconstrained
        self.log_inv_freq = nn.Parameter(init_inv_freq.log())  # shape (rotary_dim//2,)

        # precompute position indices once (0..max_pos-1)
        pos = torch.arange(max_pos, device=device).float()  # (max_pos,)
        self.register_buffer("pos", pos)

    def build_sin_cos_for_grid(self, H, W):
        # produce sin, cos of shape (H, W, rotary_dim)
        # inv_freq shape --> (rotary_dim//2,)
        inv_freq = self.log_inv_freq.exp()  # (rotary_dim//2,)
        # outer products for x and y
        px = torch.einsum("p,d->pd", self.pos[:W], inv_freq)  # (W, dim/2)
        py = torch.einsum("p,d->pd", self.pos[:H], inv_freq)  # (H, dim/2)
        # sin/cos
        sin_x = torch.sin(px)  # (W, dim/2)
        cos_x = torch.cos(px)
        sin_y = torch.sin(py)  # (H, dim/2)
        cos_y = torch.cos(py)
        # combine halves [y | x]
        # final interleave expects rotary_dim dims; we place y first then x (like earlier)
        sin = torch.zeros(H, W, self.rotary_dim, device=sin_x.device, dtype=sin_x.dtype)
        cos = torch.zeros_like(sin)
        dim_half = self.rotary_dim // 2
        sin[:, :, :dim_half] = sin_y.unsqueeze(1).expand(-1, W, -1)
        sin[:, :, dim_half:] = sin_x.unsqueeze(0).expand(H, -1, -1)
        cos[:, :, :dim_half] = cos_y.unsqueeze(1).expand(-1, W, -1)
        cos[:, :, dim_half:] = cos_x.unsqueeze(0).expand(H, -1, -1)
        return sin, cos  # (H, W, rotary_dim)

    # reuse rotate_every_two and apply_rotary_pos_emb from prior code
    def forward_build(self, H, W):
        return self.build_sin_cos_for_grid(H, W)

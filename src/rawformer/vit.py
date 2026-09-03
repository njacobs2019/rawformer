"""
ViT implementation (not true to ViT paper)
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .position_encoding import PositionScheme, apply_rope
from .types import RoPECache, Tokens


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        qkv_bias: bool,
        mlp_hidden_dim: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()

        embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout

        # MHA Layers
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.mha_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mha_dropout = nn.Dropout(p=dropout)

        # Norm layers
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden_dim, embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )

    def forward(
        self, x: Float[Tensor, "b l d"], rope_cache: RoPECache | None
    ) -> Float[Tensor, "b l d"]:
        # residual block 1
        x = x + self._mha(self.norm1(x), rope_cache)

        # residual block 2
        return x + self.mlp(self.norm2(x))

    def _mha(
        self, x: Float[Tensor, "b l dim"], rope_cache: RoPECache | None
    ) -> Float[Tensor, "b l dim"]:
        # Linear projection
        Q = self.W_q(x)  # (b l dim)
        K = self.W_k(x)  # (b l dim)
        V = self.W_v(x)  # (b l dim)

        # Rearrange
        Q = rearrange(Q, "b l (h d_head) -> b h l d_head", h=self.num_heads)
        K = rearrange(K, "b l (h d_head) -> b h l d_head", h=self.num_heads)
        V = rearrange(V, "b l (h d_head) -> b h l d_head", h=self.num_heads)

        # Optionally rotate
        if rope_cache is not None:
            Q = apply_rope(Q, rope_cache)
            K = apply_rope(K, rope_cache)

        attn_out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_out = rearrange(attn_out, "b h l d_head -> b l (h d_head)")
        attn_proj = self.mha_out_proj(attn_out)
        return self.mha_dropout(attn_proj)


class ClassToken(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.tok = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def prepend(
        self,
        tokens: Float[Tensor, "b l d"],
        rope_cache: tuple[Float[Tensor, "l rot_dim"], Float[Tensor, "l rot_dim"]]
        | None,
    ) -> tuple[
        Float[Tensor, "b l+1 d"],
        tuple[Float[Tensor, "l+1 rot_dim"], Float[Tensor, "l+1 rot_dim"]] | None,
    ]:
        cls_tok = self.tok.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        if rope_cache is not None:
            sin, cos = rope_cache
            sin = F.pad(sin, (0, 0, 1, 0), value=0.0)
            cos = F.pad(cos, (0, 0, 1, 0), value=1.0)
            rope_cache = (sin, cos)
        return tokens, rope_cache

    def extract(self, tokens: Tokens) -> Float[Tensor, "b dim"]:
        return tokens[:, 0]


class ViT(nn.Module):
    def __init__(
        self,
        patch_embed: nn.Module,
        position: PositionScheme,
        mlp_head: nn.Module | None,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden_dim: int,
        use_cls: bool,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        """
        Args:
           patch_embed: Parses and embeds img into tokens also returns grid_size
           position: RoPE or LearnedPositionEmbeddings
           mlp_head: Sub-network that operates on transformer output e.g. (b, len, dim)

           num_layers: Number of transformer blocks
           num_heads: Number of attention heads per block
           head_dim: Embedding dim per head
           mlp_hidden_dim: Hidden dimension of the per-block MLP.
           use_cls: Use class token
           qkv_bias: Whether the Q/K/V projections include a bias term.
           dropout: Probability of dropout during training
           attn_dropout: Probability of dropout during training of attention scores
               Note, used in BERT.
        """

        super().__init__()

        embed_dim = num_heads * head_dim
        self._validate(patch_embed, position, num_heads, head_dim)

        self.patch_embed = patch_embed
        self.pos_embed = position
        self.head = mlp_head
        self.cls_tok = ClassToken(embed_dim=embed_dim) if use_cls else None

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    qkv_bias=qkv_bias,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

    @staticmethod
    def _validate(
        patch_embed: nn.Module,
        position: PositionScheme,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Reject mismatched components at construction rather than at first forward.

        Each check is skipped when the component doesn't advertise the attribute, so
        custom patch embeddings and position schemes keep working.
        """
        embed_dim = num_heads * head_dim

        patch_dim = getattr(patch_embed, "embed_dim", None)
        if patch_dim is not None and patch_dim != embed_dim:
            msg = (
                f"patch_embed produces {patch_dim}-dim tokens but this ViT is built "
                f"for num_heads * head_dim = {num_heads} * {head_dim} = {embed_dim}"
            )
            raise ValueError(msg)

        rotary_dim = getattr(position, "rotary_dim", None)
        if rotary_dim is not None and rotary_dim > head_dim:
            msg = (
                f"rotary_dim {rotary_dim} exceeds head_dim {head_dim}. RoPE is "
                f"applied per head, so rotary_dim must be <= head_dim -- note this "
                f"is head_dim, not embed_dim ({embed_dim})."
            )
            raise ValueError(msg)

        n_axes = getattr(position, "n_axes", None)
        grid_rank = getattr(patch_embed, "grid_rank", None)
        if n_axes is not None and grid_rank is not None and n_axes != grid_rank:
            msg = (
                f"position scheme expects n_axes={n_axes} but patch_embed produces "
                f"a {grid_rank}D grid"
            )
            raise ValueError(msg)

    def no_weight_decay(self) -> set[str]:
        """Parameter names that must be excluded from weight decay.

        Covers the class token and every parameter of the position scheme (learned
        tables, and a learnable RoPE base if one was opted into). A learnable
        ``log_base`` gets almost no gradient signal but full weight decay, which
        drags it toward 0 and collapses every rotary frequency to ~1.0, silently
        destroying positional resolution. ``cls_tok.tok`` and the learned table are
        both ndim-3, so the common ``p.ndim >= 2 -> decay`` rule catches them too.

        Usage::

            skip = model.no_weight_decay()
            params = list(model.named_parameters())
            optim = torch.optim.AdamW(
                [
                    {"params": [p for n, p in params if n not in skip],
                     "weight_decay": 0.05},
                    {"params": [p for n, p in params if n in skip],
                     "weight_decay": 0.0},
                ],
                lr=1e-3,
            )
        """
        names = {n for n, _ in self.named_parameters() if n.startswith("pos_embed.")}
        if self.cls_tok is not None:
            names.add("cls_tok.tok")
        return names

    def forward(self, x: Float[Tensor, "b c h w"]) -> Tensor:
        # Create and embed patches into tokens
        x, grid_size = self.patch_embed(x)

        # Prepare position embeddings
        x, rope_cache = self.pos_embed.prepare(x, grid_size)

        if self.cls_tok is not None:
            x, rope_cache = self.cls_tok.prepend(x, rope_cache)

        for layer in self.layers:
            x = layer(x, rope_cache)

        x = self.norm(x)

        if self.cls_tok is not None:
            x = self.cls_tok.extract(x)

        if self.head is not None:
            x = self.head(x)

        return x

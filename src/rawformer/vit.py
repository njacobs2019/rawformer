"""
ViT implementation (not true to ViT paper)
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .position_encoding import RoPE2D, RoPECache, apply_rope


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        qkv_bias: bool,
        mlp_hidden_dim: int,
        dropout: float,
        attn_dropout: float,
        attn_mask: Tensor | None,
    ) -> None:
        super().__init__()

        embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.attn_mask = attn_mask
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
        self, x: Float[Tensor, "b l d"], rope_cache: RoPECache | None = None
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

        # Optionally rotate
        if rope_cache is not None:
            Q = apply_rope(Q, rope_cache)
            K = apply_rope(K, rope_cache)

        # Rearrange
        Q = rearrange(Q, "b l (h d_head) -> b h l d_head", h=self.num_heads)
        K = rearrange(K, "b l (h d_head) -> b h l d_head", h=self.num_heads)
        V = rearrange(V, "b l (h d_head) -> b h l d_head", h=self.num_heads)

        attn_out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=self.attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_out = rearrange(attn_out, "b h l d_head -> b l (h d_head)")
        attn_proj = self.mha_out_proj(attn_out)
        return self.mha_dropout(attn_proj)


class ViT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        qkv_bias: bool,
        patch_embedding: nn.Module,
        rope: RoPE2D | None,  # TODO: Implement
        position_embedding: nn.Module | None,
        use_cls_tok: bool,
        head: nn.Module | None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        attn_mask: Tensor | None = None,  # TODO: Implement and register as buffer
    ) -> None:
        """Vision Transformer

        Composes a patch embedding, a stack of transformer blocks, a positional
        scheme (either RoPE applied inside attention, or an additive position
        embedding on the token sequence), and a head (either a CLS token head
        or a per-token output head).

        Args:
            num_layers: Number of transformer blocks in the stack.
            num_heads: Number of attention heads per block. Must divide `embed_dim`.
            embed_dim: Width of the residual stream and token embeddings. Per-head
                dimension is `embed_dim // num_heads`.
            mlp_hidden_dim: Hidden dimension of the per-block MLP.
            qkv_bias: Whether the Q/K/V projections include a bias term. Original
                ViT uses `True`; many modern LLMs use `False`.
            patch_embedding: Module mapping an input image of shape `(B, C, H, W)`
                to a token sequence of shape `(B, L, embed_dim)`.
            rope: Rotary position embedding applied to Q and K inside attention.
                Mutually exclusive with `position_embedding`.
            position_embedding: Additive position embedding applied to the token
                sequence before the transformer stack. Mutually exclusive with
                `rope`.
            use_cls_tok: Use a class token (prepend at beginning, extract at end)
            head: Sub-network that operates on transformer output e.g. (b, len, dim)
            dropout: Probability of dropout during training
            attn_dropout: Probability of dropout during training of attention scores
                Note, used in BERT.
            attn_mask: Attention mask

        Raises:
            ValueError: If both or neither of `rope` and `position_embedding` are
                provided.
            ValueError: If both or neither of `cls_head` and `out_head` are
                provided.
        """
        super().__init__()

        # VALIDATE INPUTS
        if (rope is None) == (position_embedding is None):
            msg = (
                "Provide exactly one of `rope` or `position_embedding` "
                f"(got rope={rope is not None}, "
                f"position_embedding={position_embedding is not None})"
            )
            raise ValueError(msg)

        if embed_dim % num_heads != 0:
            msg = (
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(msg)

        self.patch_embedding = patch_embedding

        self.pos_embedding = position_embedding
        self.rope = rope

        self.head = head
        self.cls_tok = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_tok else None
        )

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    head_dim=embed_dim // num_heads,
                    qkv_bias=qkv_bias,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    attn_mask=attn_mask,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x: Float[Tensor, "b c h w"]) -> Tensor:
        # Get RoPE rot matrices
        _b, _c, h, w = x.shape
        rope_cache = None
        if self.rope is not None:
            rope_cache = self.rope.build_cache(h, w, dtype=x.dtype)

        # Create patch embeddings
        x = self.patch_embedding(x)  # (b, len, embed_dim)

        # Prepend class token
        if self.cls_tok is not None:
            cls_tok = self.cls_tok.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tok, x), dim=1)

        # Position embedding
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)  # (b, len, embed_dim)
            x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, rope_cache)

        x = self.norm(x)

        # Extract class token
        if self.cls_tok is not None:
            x = x[:, 0, :]

        # Run output head
        if self.head is not None:
            x = self.head(x)

        return x


class SimplePatchEmbedding(nn.Module):
    """
    Creates non-overlapping patches and linearly embeds them
    """

    def __init__(self, patch_len: int, channels: int, embed_dim: int) -> None:
        super().__init__()

        self.patch_len = patch_len
        self.fc = nn.Linear(patch_len * patch_len * channels, embed_dim)

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b len embed_dim"]:
        _b, _c, h, w = x.shape

        assert h % self.patch_len == 0, "Height must be divisible by patch_len"
        assert w % self.patch_len == 0, "Width must be divisible by patch_len"

        tokens = F.unfold(x, kernel_size=self.patch_len, stride=self.patch_len)
        tokens = rearrange(tokens, "b dim len -> b len dim")

        return self.fc(tokens)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()

        self.max_len = max_len
        self.E = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(1, max_len, embed_dim))
        )  # Same as ViT and BERT

    def forward(self, x: Float[Tensor, "b l d"]) -> Float[Tensor, "b l d"]:
        length = x.shape[1]
        assert length <= self.max_len

        return x + self.E[:, :length, :]

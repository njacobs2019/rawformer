"""
My ViT Implementation, note the following changes from Attention is All You Need
- Attention is all you need --> ViT
- $d_{model}$ --> embed_dim
- $d_{ff}$ --> mlp_hidden_dim
"""

import math

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


@beartype
def scaled_dot_product_attn(
    Q: Float[Tensor, "b l d_k"],
    K: Float[Tensor, "b l d_k"],
    V: Float[Tensor, "b l d_v"],
) -> Float[Tensor, "b l d_v"]:
    # TODO: Add attention mask
    d_k = K.shape[-1]

    factor = 1 / math.sqrt(d_k)

    K_t = torch.transpose(K, 1, 2)  # (b, d_k, l)
    scores = torch.bmm(Q, K_t) * factor  # (b, l, l)
    attn_weights = torch.softmax(scores, dim=-1)  # (b, l, l)

    return torch.bmm(attn_weights, V)


@beartype
class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, d_k: int, d_v: int, qkv_bias: bool) -> None:
        super().__init__()
        self.W_q = nn.Linear(embed_dim, d_k, bias=qkv_bias)
        self.W_k = nn.Linear(embed_dim, d_k, bias=qkv_bias)
        self.W_v = nn.Linear(embed_dim, d_v, bias=qkv_bias)

    def forward(self, X: Float[Tensor, "b l embed_dim"]) -> Float[Tensor, "b l d_v"]:

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        return scaled_dot_product_attn(Q, K, V)


@beartype
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        d_k: int,
        d_v: int,
        qkv_bias: bool,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(embed_dim=embed_dim, d_k=d_k, d_v=d_v, qkv_bias=qkv_bias)
                for _ in range(num_heads)
            ]
        )

        self.out_proj = nn.Linear(num_heads * d_v, embed_dim)

    def forward(
        self, x: Float[Tensor, "b l embed_dim"]
    ) -> Float[Tensor, "b l embed_dim"]:
        attn_head_out = torch.cat([head(x) for head in self.heads], dim=2)
        return self.out_proj(attn_head_out)


@beartype
class ParallelMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        d_k: int,
        d_v: int,
        qkv_bias: bool,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.W_q = nn.Linear(embed_dim, d_k * num_heads, bias=qkv_bias)
        self.W_k = nn.Linear(embed_dim, d_k * num_heads, bias=qkv_bias)
        self.W_v = nn.Linear(embed_dim, d_v * num_heads, bias=qkv_bias)

        self.out_proj = nn.Linear(num_heads * d_v, embed_dim)

    def forward(
        self, x: Float[Tensor, "batch length embed_dim"]
    ) -> Float[Tensor, "batch length embed_dim"]:

        Q = self.W_q(x)
        Q = rearrange(Q, "b l (d h) -> (b h) l d", h=self.num_heads)

        K = self.W_k(x)
        K = rearrange(K, "b l (d h) -> (b h) l d", h=self.num_heads)

        V = self.W_v(x)
        V = rearrange(V, "b l (d h) -> (b h) l d", h=self.num_heads)

        attn_out = scaled_dot_product_attn(Q, K, V)
        attn_merged = rearrange(attn_out, "(b h) l d -> b l (h d)", h=self.num_heads)

        return self.out_proj(attn_merged)


@beartype
class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim, bias=True)
        self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim, bias=True)

    def forward(
        self, x: Float[Tensor, "b l embed_dim"]
    ) -> Float[Tensor, "b l embed_dim"]:
        x = self.fc1(x)
        x = F.gelu(x)
        return self.fc2(x)


@beartype
class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        d_k: int,
        d_v: int,
        qkv_bias: bool,
        mlp_hidden_dim: int,
    ) -> None:
        # TODO: Add dropout_prob
        super().__init__()

        # First residual block
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mha = ParallelMultiHeadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            d_k=d_k,
            d_v=d_v,
            qkv_bias=qkv_bias,
        )

        # Second residual block
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim)

    def forward(self, x: Float[Tensor, "b l d"]) -> Float[Tensor, "b l d"]:
        # Residual block 1
        x = x + self.mha(self.norm1(x))

        # Residual block 2
        return x + self.mlp(self.norm2(x))


@beartype
class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        super().__init__()

        self.max_len = max_seq_len + 1
        self.E = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(1, self.max_len, embed_dim))
        )  # Same as ViT and BERT

    def forward(self, x: Float[Tensor, "b l d"]) -> Float[Tensor, "b l d"]:
        length = x.shape[1]
        assert length <= self.max_len

        return x + self.E[:, :length, :]


@beartype
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


@beartype
class ViT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        d_k: int,
        d_v: int,
        qkv_bias: bool,
        mlp_hidden_dim: int,
        patch_embedding: nn.Module,
        position_embedding: nn.Module,
        head: nn.Module | None,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    d_k=d_k,
                    d_v=d_v,
                    qkv_bias=qkv_bias,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.patch_embedding = patch_embedding

        self.position_embedding = position_embedding

        self.cls_tok = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )  # same as ViT and BERT initialization

        self.head = head or nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: Float[Tensor, "b c h w"]):

        # Create patch embeddings
        x = self.patch_embedding(x)  # (b, len, embed_dim)

        # Prepend cls token
        # x = torch.cat([])

        # patchify the tokens
        # cat the class token in
        # add position embeddings

        # run through layers

        # Extract cls token
        # run cls token through MLP


# TODO: FIX DROPOUT
# TODO: When to turn off bias in linear layers?

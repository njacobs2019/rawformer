"""
My implementation of the encoder from Attention is all you need
"""

import math

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn


@beartype
def scaled_dot_product_attn(
    Q: Float[Tensor, "b l d_k"],
    K: Float[Tensor, "b l d_k"],
    V: Float[Tensor, "b l d_v"],
) -> Float[Tensor, "b l d_v"]:
    # NOTE: This is only 1 head
    _batch, _length, d_k = K.shape

    factor = 1 / math.sqrt(d_k)

    K_t = torch.transpose(K, 1, 2)  # (b, d_k, l)
    scaled_scores = torch.bmm(Q, K_t) * factor  # (b, l, l)
    attn_weights = torch.softmax(scaled_scores, dim=2)  # (b, l, l)

    return torch.bmm(attn_weights, V)


@beartype
class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, qkv_bias: bool) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_v, bias=qkv_bias)

    def forward(self, X: Float[Tensor, "b l d_model"]) -> Float[Tensor, "b l d_v"]:

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        return scaled_dot_product_attn(Q, K, V)


@beartype
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Float[Tensor, "b l d_model"]) -> Float[Tensor, "b l d_model"]:
        x = self.layer1(x)
        x = F.relu(x)
        return self.layer2(x)


@beartype
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        dropout_prob: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_k, d_v, qkv_bias) for x in range(num_heads)]
        )

        self.multihead_out = nn.Linear(num_heads * d_v, d_model, bias=False)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)

        self.ff = FFN(d_model, d_ff)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Float[Tensor, "b l d_model"]) -> Float[Tensor, "b l d_model"]:

        head_results_list = [head(x) for head in self.heads]  # list of (b, l, d_v)
        head_results = torch.cat(head_results_list, dim=2)
        mha_out = self.multihead_out(head_results)
        x = self.norm1(x + self.dropout(mha_out))

        return self.norm2(x + self.dropout(self.ff(x)))


@beartype
class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        dropout_prob: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, d_k, d_v, dropout_prob, qkv_bias)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Float[Tensor, "b l d_model"]) -> Float[Tensor, "b l d_model"]:
        for layer in self.layers:
            x = layer(x)

        return x

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(config, key: str, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class CausalSelfAttention(nn.Module):
    """Causal self-attention block used by the SAFE-AR backbone."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "silu":
            x = F.silu(x)
        else:
            x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = FeedForward(hidden_size, intermediate_size, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SafeCausalDiTLM(nn.Module):
    """DDG-style autoregressive causal transformer for SAFE token generation."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(_get(config, "hidden_size", 768))
        self.max_position_embeddings = int(
            _get(config, "max_position_embeddings", _get(config, "length", 256))
        )
        num_heads = int(_get(config, "num_attention_heads", _get(config, "n_heads", 12)))
        num_layers = int(_get(config, "num_hidden_layers", _get(config, "n_blocks", 12)))
        intermediate_size = int(_get(config, "intermediate_size", 4 * self.hidden_size))
        dropout = float(_get(config, "hidden_dropout_prob", _get(config, "dropout", 0.1)))
        activation = str(_get(config, "hidden_act", "gelu"))
        layer_norm_eps = float(_get(config, "layer_norm_eps", 1e-5))

        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        if bool(_get(config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        initializer_range = float(_get(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        sigma: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        x_emb: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        del sigma, cond

        seq_len = input_ids.shape[1]
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Input length {seq_len} exceeds model limit "
                f"{self.max_position_embeddings}."
            )

        if x_emb is None:
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(input_ids.shape[0], seq_len)
            x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
            x = self.dropout(x)
        else:
            x = x_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)

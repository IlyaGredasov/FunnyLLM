from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm1(x)
        attention_out, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + attention_out
        x = x + self.mlp(self.layer_norm2(x))
        return x


class FunnyLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.layer_norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.drop(x)
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.layer_norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len :]
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < vals[:, [-1]], float("-inf"), logits)
            probabilities = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CausalTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.d_model = model_cfg["d_model"]
        self.n_heads = model_cfg["n_heads"]
        self.n_layers = model_cfg["n_layers"]
        self.d_ff = model_cfg["d_ff"]
        self.dropout = model_cfg["dropout"]
        self.max_seq_len = model_cfg["max_seq_len"]
        self.vocab_size = model_cfg["vocab_size"]
        # C-03: 底因基底投影 π: d_model -> base_dim，所有几何距离在此空间计算
        self.base_dim = model_cfg.get("base_dim", None)
        if self.base_dim is not None:
            self.base_projection = nn.Linear(self.d_model, self.base_dim)
        else:
            self.base_projection = None
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len, self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)
        else:
            pad_mask = None
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        logits = self.lm_head(x)
        # C-03: 返回投影到底因基底 B 的隐含态，供几何距离/度量使用
        if self.base_projection is not None:
            hidden = self.base_projection(x)
        else:
            hidden = x
        return logits, hidden

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            _, hidden = self.forward(input_ids, attention_mask)
        return hidden

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        # 注(C-12/B-05): 当前为标准的配分函数温度采样(B-03)。
        # 论文要求 λ→∞ 时取测地线(推理=平行移动)，该约束解码尚未实现，
        # 仅作启发式元语言 + 正则化思想使用，不可前移为严格数学理论。
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_ids.size(1) >= self.max_seq_len:
                    break
                logits, _ = self.forward(input_ids)
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)
                if top_k > 0:
                    top_k_logits, _ = torch.topk(next_logits, top_k)
                    min_val = top_k_logits[:, -1:].expand_as(next_logits)
                    next_logits = torch.where(
                        next_logits < min_val,
                        torch.full_like(next_logits, float("-inf")),
                        next_logits,
                    )
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
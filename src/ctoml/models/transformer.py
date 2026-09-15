import math

import torch
from torch import nn


class MathAttention(nn.Module):


    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by nhead")
        self.heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, padding_mask, causal=False):
        batch, length, width = query.shape
        def split(x):
            return x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(memory)), split(self.v(memory))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        mask = padding_mask[:, None, None, :]
        if causal:
            mask = mask | torch.ones(length, memory.shape[1], dtype=torch.bool, device=query.device).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)

        attention = self.dropout(scores.softmax(dim=-1))
        attended = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class EncoderLayer(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.attention = MathAttention(width, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attention(x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.self_attention = MathAttention(width, heads, dropout)
        self.cross_attention = MathAttention(width, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(width) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, target_mask, causal=True)))
        token_features = self.cross_attention(x, memory, source_mask)
        x = self.norm2(x + self.dropout(token_features))
        return self.norm3(x + self.dropout(self.ffn(x))), token_features


class CtoMLModel(nn.Module):
    def __init__(self, feature_dim, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.3, max_length=2048):
        super().__init__()
        if min(feature_dim, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, max_length) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        self.vocab_size, self.d_model = vocab_size, d_model
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        positions = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:d_model // 2])
        self.register_buffer("positions", encoding, persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features, feature_lengths, decoder_inputs):
        if features.ndim != 3 or decoder_inputs.ndim != 2 or feature_lengths.shape != (features.shape[0],):
            raise ValueError("Expected features [B,T,F], lengths [B], decoder_inputs [B,L]")
        if decoder_inputs.shape[0] != features.shape[0] or decoder_inputs.shape[1] == 0:
            raise ValueError("Decoder must have matching batch size and at least one input token")
        if (feature_lengths < 1).any() or (feature_lengths > features.shape[1]).any():
            raise ValueError("Feature lengths must be in [1,T]")
        if (decoder_inputs[:, 0] == 0).any():
            raise ValueError("First decoder input must be non-padding (normally BOS)")
        if max(features.shape[1], decoder_inputs.shape[1]) > self.positions.shape[0]:
            raise ValueError("Sequence exceeds model max_length")
        source_mask = torch.arange(features.shape[1], device=features.device)[None, :] >= feature_lengths[:, None]
        target_mask = decoder_inputs.eq(0)
        memory = self.dropout(self.feature_projection(features) + self.positions[:features.shape[1]])
        for layer in self.encoder:
            memory = layer(memory, source_mask)
        decoded = self.dropout(self.embedding(decoder_inputs) * math.sqrt(self.d_model) + self.positions[:decoder_inputs.shape[1]])
        for layer in self.decoder:
            decoded, token_features = layer(decoded, memory, source_mask, target_mask)
        return {"logits": self.classifier(decoded), "token_features": token_features}

    @torch.no_grad()
    def generate(self, features, feature_lengths, max_length=64, bos_id=1, eos_id=2, pad_id=0):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full((features.shape[0], 1), bos_id, dtype=torch.long, device=features.device)
            done = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
            for _ in range(max_length):
                logits = self(features, feature_lengths, tokens)["logits"][:, -1].clone()
                logits[:, [pad_id, bos_id]] = -torch.inf
                next_token = logits.argmax(-1).masked_fill(done, pad_id)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                done = done | next_token.eq(eos_id)
                if done.all():
                    break
            return tokens[:, 1:]
        finally:
            self.train(was_training)

import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.fc_out(out)


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, heads=4, block_size=64):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.attn = SelfAttention(embed_size, heads)
        self.ln = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.attn(x)
        x = self.ln(x)
        return self.fc(x)
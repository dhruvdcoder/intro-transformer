import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention using PyTorch's scaled_dot_product_attention.
    Supports past key/value caching for fast autoregressive decoding.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        # x: (batch, seq_len, d_model)
        B, T, C = x.size()
        # linear projections and reshape for multi-head
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # append past if provided (for caching)
        if past_kv is not None:
            past_k, past_v = past_kv  # each: (B, num_heads, past_T, head_dim)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # combine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_out)
        # return output and present kv for caching
        present_kv = (k, v)
        return out, present_kv


class FeedForward(nn.Module):
    """Simple position-wise feed-forward network."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block: self-attention + feed-forward with layer norm and residuals."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None):
        # Self-attention
        residual = x
        x_norm = self.ln1(x)
        attn_out, present_kv = self.attn(x_norm, past_kv)
        x = residual + self.dropout(attn_out)

        # Feed-forward
        residual = x
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = residual + self.dropout(ff_out)
        return x, present_kv


class TinyTransformer(nn.Module):
    """
    Transformer model with absolute (learned) positional embeddings,
    KV caching, and a decode() method for autoregressive generation.
    """
    def __init__(
        self,
        vocab_size,
        max_seq_len=512,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        dropout=0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: (batch, seq_len)
        B, T = x.size()
        assert T <= self.max_seq_len, "Sequence too long"
        # embeddings
        tok = self.token_emb(x)                           # (B, T, d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos = self.pos_emb(pos)                           # (1, T, d_model)
        h = tok + pos
        # pass through transformer blocks (no caching)
        for block in self.blocks:
            h, _ = block(h, past_kv=None)
        # final norm and head
        h = self.ln(h)
        return self.head(h)                               # (B, T, vocab_size)

    def _forward_step(self, x, past_kvs):
        """
        Single-step forward for decoding. x:
          (batch, 1) tensor of latest token IDs.
        past_kvs: list of length num_layers of (past_k, past_v) tuples.
        """
        B, T = x.size()  # here T == 1
        # token + positional embeddings
        tok = self.token_emb(x)                           # (B, 1, d_model)
        # compute current position from cache length
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        positions = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0)
        pos = self.pos_emb(positions)                     # (1, 1, d_model)
        h = tok + pos
        new_past = []
        # pass through each block, keeping caches
        for block, past in zip(self.blocks, past_kvs or [None]*len(self.blocks)):
            h, present = block(h, past)
            new_past.append(present)
        h = self.ln(h)
        logits = self.head(h)                             # (B, 1, vocab_size)
        return logits, new_past

    @torch.no_grad()
    def decode(self, input_ids, max_length, eos_token_id=None):
        """
        Autoregressive decode: appends up to max_length tokens.
        input_ids: (batch, seq_len) initial context.
        """
        device = input_ids.device
        # initialize cache
        past_kvs = [None] * len(self.blocks)
        # generate tokens one at a time
        for _ in range(max_length):
            # take last token only
            x = input_ids[:, -1:].to(device)
            logits, past_kvs = self._forward_step(x, past_kvs)
            # pick next token (greedy)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids

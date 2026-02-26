import math
import torch
import torch.nn as nn


# ── Device utility ────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Model configs ─────────────────────────────────────────────────────────────

SMALL_CFG = dict(n_embd=16, n_layer=4, n_head=4)
LARGE_CFG = dict(n_embd=32, n_layer=4, n_head=4)


# ── Architecture ──────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = att.softmax(-1)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying: input embedding and output projection share weights
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x   = self.tok_emb(idx) + self.pos_emb(pos)
        x   = self.blocks(x)
        x   = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, greedy=False):
        from data import EOS_ID
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.block_size:]
            logits   = self(ids_cond)[:, -1, :]
            if greedy:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                probs   = (logits / max(temperature, 1e-8)).softmax(-1)
                next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == EOS_ID:
                break
        return ids


def build_model(model_size, block_size, vocab_size):
    cfg = SMALL_CFG if model_size == 'small' else LARGE_CFG
    return MiniGPT(vocab_size=vocab_size, block_size=block_size, **cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

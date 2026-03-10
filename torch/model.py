import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Device utility ────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Model configs ─────────────────────────────────────────────────────────────

SMALL_CFG  = dict(n_embd=16,  n_layer=4,  n_head=4)     # ~13K params
LARGE_CFG  = dict(n_embd=32,  n_layer=4,  n_head=4)     # ~52K params
MEDIUM_CFG = dict(n_embd=192, n_layer=8,  n_head=8)     # ~3.5M params
XLARGE_CFG = dict(n_embd=384, n_layer=12, n_head=12)    # ~21M params

MODEL_CONFIGS = {
    'small':  SMALL_CFG,
    'large':  LARGE_CFG,
    'medium': MEDIUM_CFG,
    'xlarge': XLARGE_CFG,
}


# ── Architecture ──────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.dropout  = dropout
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.resid_drop(self.attn(self.ln1(x)))
        x = x + self.resid_drop(self.mlp(self.ln2(x)))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.n_layer    = n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
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

    def forward(self, idx, return_hidden_states=False):
        """
        Forward pass.

        Args:
            idx: (B, T) token indices
            return_hidden_states: if True, also return a list of hidden states
                after each transformer block (for probing / mechinterp).

        Returns:
            logits: (B, T, vocab_size)
            hidden_states (optional): list of (B, T, n_embd) tensors, one per layer
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x   = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if return_hidden_states:
            return logits, hidden_states
        return logits

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


def build_model(model_size, block_size, vocab_size, dropout=0.0):
    cfg = MODEL_CONFIGS[model_size]
    return MiniGPT(vocab_size=vocab_size, block_size=block_size, dropout=dropout, **cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_state_dict_compat(model, state_dict):
    """Load state dict with backward compatibility for old checkpoints.
    Filters out legacy causal mask buffers that are no longer used (SDPA)."""
    filtered = {k: v for k, v in state_dict.items() if not k.endswith('.mask')}
    model.load_state_dict(filtered)

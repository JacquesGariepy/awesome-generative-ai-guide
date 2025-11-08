# Chapitre 6: Pré-entraînement de LLMs from Scratch

## Introduction

Le **pré-entraînement** est le processus le plus coûteux et complexe dans la création d'un LLM. C'est là que le modèle apprend les structures du langage à partir de milliards de tokens.

### Pourquoi pré-entraîner?

```python
"""
Pré-entraînement vs Fine-tuning:

Pré-entraînement:
  • Corpus: Massive (100GB-10TB de texte)
  • Objectif: Apprendre le langage général
  • Données: Non-supervisées (internet, livres, code)
  • Coût: $1M-$100M+ (GPU months)
  • Exemple: GPT-3 base model

Fine-tuning:
  • Corpus: Small (1MB-1GB)
  • Objectif: Task spécifique
  • Données: Supervisées (Q&A, instructions)
  • Coût: $10-$10k
  • Exemple: ChatGPT (GPT-3 + RLHF)

Quand pré-entraîner from scratch?
  ✅ Domain très spécifique (medical, legal, code)
  ✅ Langue rare / non couverte
  ✅ Besoins de confidentialité
  ✅ Research / innovation
  ❌ Applications générales (use pre-trained models!)

Ce chapitre: Train small GPT-2 style model (~50M params)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math


@dataclass
class ModelConfig:
    """Configuration for GPT model"""
    # Architecture
    vocab_size: int = 50257  # GPT-2 vocabulary
    n_layers: int = 6        # Number of transformer blocks
    n_heads: int = 6         # Attention heads
    n_embd: int = 384        # Embedding dimension
    block_size: int = 256    # Context window
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 20

    # Optimization
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: bool = False  # PyTorch 2.0 compile


@dataclass
class PretrainingStats:
    """Statistics for pretraining"""
    name: str
    parameters: str
    dataset_size: str
    training_tokens: str
    gpus: str
    training_time: str
    cost_estimate: str


# Famous pretraining runs
PRETRAINING_EXAMPLES = [
    PretrainingStats(
        name="GPT-3",
        parameters="175B",
        dataset_size="570GB (filtered)",
        training_tokens="300B tokens",
        gpus="~10,000 V100s",
        training_time="~1 month",
        cost_estimate="$4.6M+"
    ),
    PretrainingStats(
        name="Llama 2 70B",
        parameters="70B",
        dataset_size="2TB",
        training_tokens="2T tokens",
        gpus="~2000 A100s",
        training_time="~1 month",
        cost_estimate="~$3M"
    ),
    PretrainingStats(
        name="GPT-2",
        parameters="1.5B",
        dataset_size="40GB (WebText)",
        training_tokens="~10B tokens",
        gpus="~256 TPUv3",
        training_time="~1 week",
        cost_estimate="~$50k"
    ),
    PretrainingStats(
        name="Our Model (tutorial)",
        parameters="50M",
        dataset_size="100MB",
        training_tokens="~25M tokens",
        gpus="1x RTX 3090",
        training_time="~4 hours",
        cost_estimate="~$5"
    )
]


def print_pretraining_comparison():
    """Print comparison of pretraining runs"""
    print("="*100)
    print("FAMOUS PRETRAINING RUNS")
    print("="*100)

    print(f"\n{'Model':<20} {'Parameters':<12} {'Dataset':<20} {'Tokens':<15} {'Cost':<12}")
    print("-"*100)

    for stats in PRETRAINING_EXAMPLES:
        print(f"{stats.name:<20} {stats.parameters:<12} {stats.dataset_size:<20} "
              f"{stats.training_tokens:<15} {stats.cost_estimate:<12}")

    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print("""
1. Scaling Laws (Kaplan et al., 2020):
   • Performance scales predictably with:
     - Model size (parameters)
     - Dataset size (tokens)
     - Compute (FLOPs)

2. Chinchilla Optimal (Hoffmann et al., 2022):
   • For compute budget C, optimal split:
     - Parameters: N ∝ C^0.5
     - Tokens: D ∝ C^0.5
   • Most models are overtrained (too many params, too few tokens)
   • Llama 2 follows this: 70B params, 2T tokens

3. Our tutorial model:
   • Small enough to train on 1 GPU
   • Big enough to generate coherent text
   • ~50M params, ~25M tokens
    """)


if __name__ == "__main__":
    print_pretraining_comparison()
```

## 1. Architecture GPT

```python
"""
GPT Architecture = Decoder-only Transformer

Key components:
  • Token + Position embeddings
  • N transformer blocks
  • Final layer norm
  • LM head (project to vocabulary)

Differences from BERT:
  • Causal attention (can't see future)
  • No encoder (decoder-only)
  • Autoregressive generation
"""

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention (GPT-style)

    Attention with causal mask: token i can only attend to tokens ≤ i
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.n_embd % config.n_heads == 0

        # Q, K, V projections for all heads (batched)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Causal mask (lower triangular)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch, seq_len, n_embd)

        Returns:
            Output (batch, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch, seq_len, n_embd

        # Calculate Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Split into heads: (B, T, C) → (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Attention: (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum: (B, n_heads, T, head_size)
        y = att @ v

        # Concatenate heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """
    Feed-forward network (MLP)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block (GPT-2 style)

    Pre-LN: LayerNorm before attention and MLP
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Attention with residual
        x = x + self.attn(self.ln_1(x))

        # MLP with residual
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    """
    GPT model (decoder-only transformer)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (share embeddings and output weights)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Count parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            idx: Input token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (optional)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Position IDs
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t,)

        # Embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # (b, t, vocab_size)

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens

        Args:
            idx: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if specified)

        Returns:
            Generated tokens (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else \
                      idx[:, -self.config.block_size:]

            # Forward
            logits, _ = self(idx_cond)

            # Focus on last token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling (optional)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Demo
if __name__ == "__main__":
    print("="*80)
    print("GPT ARCHITECTURE")
    print("="*80)

    # Create small model
    config = ModelConfig(
        vocab_size=50257,
        n_layers=4,
        n_heads=4,
        n_embd=256,
        block_size=128
    )

    model = GPT(config)

    print(f"\nArchitecture:")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Context window: {config.block_size}")

    # Test forward pass
    batch_size = 2
    seq_len = 32

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(x, targets=x)

    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item() if loss is not None else 'N/A'}")

    # Test generation
    print("\n--- Generation Test ---")
    start_ids = torch.randint(0, config.vocab_size, (1, 1))
    generated = model.generate(start_ids, max_new_tokens=10)

    print(f"Start tokens: {start_ids.tolist()}")
    print(f"Generated: {generated.tolist()}")
```

*[Suite avec Data Pipeline et Training Loop dans la partie 2...]*

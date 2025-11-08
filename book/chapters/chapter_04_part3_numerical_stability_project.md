# Chapitre 4 (Partie 3): Numerical Stability et Projet Complet

## 5. Numerical Stability

```python
"""
Numerical Stability = Prévenir NaN, Inf, et autres catastrophes numériques

Problèmes courants:
  - Overflow: nombres trop grands (> 10^38 en FP32)
  - Underflow: nombres trop petits (< 10^-38)
  - Loss of precision: opérations sur nombres très différents
  - Gradient explosion/vanishing

Solutions:
  - Log-sum-exp trick
  - Gradient clipping
  - Layer normalization
  - Mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NumericalStability:
    """
    Techniques for numerical stability in LLMs
    """

    @staticmethod
    def overflow_underflow_demo():
        """
        Demonstrate overflow and underflow issues
        """
        print("="*80)
        print("OVERFLOW AND UNDERFLOW")
        print("="*80)

        print("""
Float32 (FP32) range:
  • Max: ~3.4 × 10^38
  • Min (positive): ~1.2 × 10^-38
  • Precision: ~7 decimal digits

Problems:

1. Overflow (number too large)
   exp(100) = 2.7 × 10^43 > max → inf

2. Underflow (number too small)
   exp(-100) = 3.7 × 10^-44 < min → 0

3. Loss of precision
   1e10 + 1 = 1e10 (1 is lost due to precision limits)
        """)

        # Overflow example
        x = torch.tensor([100.0])
        exp_x = torch.exp(x)
        print(f"\nOverflow example:")
        print(f"  exp(100) = {exp_x.item()}")
        print(f"  → {exp_x.item():.2e} (infinity!)")

        # Underflow example
        y = torch.tensor([-100.0])
        exp_y = torch.exp(y)
        print(f"\nUnderflow example:")
        print(f"  exp(-100) = {exp_y.item()}")
        print(f"  → {exp_y.item():.2e} (essentially zero)")

        # Precision loss
        large = torch.tensor([1e10])
        small = torch.tensor([1.0])
        result = large + small
        print(f"\nPrecision loss:")
        print(f"  1e10 + 1 = {result.item()}")
        print(f"  Lost: {(large + small - large).item()} (should be 1)")

    @staticmethod
    def logsumexp_trick():
        """
        Log-sum-exp trick for numerical stability
        """
        print("\n" + "="*80)
        print("LOG-SUM-EXP TRICK")
        print("="*80)

        print("""
Problem: Computing log(Σ exp(x_i)) overflows

Example:
  x = [100, 101, 102]
  Σ exp(x) = exp(100) + exp(101) + exp(102)
           → All overflow to inf!

Solution: Log-sum-exp trick
  log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))

Steps:
  1. Find max(x)
  2. Subtract max from all x_i
  3. Compute exp, sum, log
  4. Add back max

Why it works:
  • exp(x_i - max) ≤ exp(0) = 1 (no overflow!)
  • At least one term = exp(0) = 1 (no underflow)
  • Mathematically equivalent

Used in:
  • Softmax computation
  • Log-probabilities
  • Attention (log-space for numerical stability)
        """)

        # Unstable version
        x = torch.tensor([100.0, 101.0, 102.0])

        # Naive (overflows)
        try:
            sum_exp = torch.sum(torch.exp(x))
            log_sum_exp_naive = torch.log(sum_exp)
            print(f"\nNaive approach:")
            print(f"  x = {x.tolist()}")
            print(f"  Σ exp(x) = {sum_exp.item()}")
            print(f"  log(Σ exp(x)) = {log_sum_exp_naive.item()}")
        except:
            print("  → Overflow!")

        # Stable version (log-sum-exp)
        max_x = torch.max(x)
        log_sum_exp_stable = max_x + torch.log(torch.sum(torch.exp(x - max_x)))

        print(f"\nLog-sum-exp trick:")
        print(f"  max(x) = {max_x.item()}")
        print(f"  x - max = {(x - max_x).tolist()}")
        print(f"  exp(x - max) = {torch.exp(x - max_x).tolist()}")
        print(f"  log(Σ exp(x - max)) = {torch.log(torch.sum(torch.exp(x - max_x))).item():.4f}")
        print(f"  Result: {log_sum_exp_stable.item():.4f}")

        # PyTorch built-in
        log_sum_exp_torch = torch.logsumexp(x, dim=0)
        print(f"\nPyTorch logsumexp: {log_sum_exp_torch.item():.4f}")
        print(f"  (uses log-sum-exp trick internally)")

    @staticmethod
    def gradient_clipping():
        """
        Gradient clipping to prevent exploding gradients
        """
        print("\n" + "="*80)
        print("GRADIENT CLIPPING")
        print("="*80)

        print("""
Problem: Exploding Gradients
  • In deep networks, gradients can grow exponentially
  • Causes unstable training (loss → NaN)
  • Common in RNNs, deep Transformers

Solution: Gradient Clipping
  Limit gradient magnitude

Two methods:

1. Clip by Value
   if |g| > threshold:
       g = threshold × sign(g)

2. Clip by Norm (most common)
   total_norm = √(Σ ||g_i||²)
   if total_norm > max_norm:
       g_i = g_i × (max_norm / total_norm)

Typical max_norm: 1.0 to 5.0

Used in:
  • All LLM training (GPT, BERT, etc.)
  • RNNs, LSTMs
  • Prevents training collapse
        """)

        # Example
        torch.manual_seed(42)

        # Model with large gradients
        x = torch.randn(10, 5, requires_grad=True)
        W = torch.randn(5, 3, requires_grad=True) * 100  # Large weights

        # Forward
        y = torch.matmul(x, W)
        loss = y.sum()

        # Backward
        loss.backward()

        # Check gradient norms
        grad_x_norm = torch.norm(x.grad)
        grad_W_norm = torch.norm(W.grad)

        print(f"\nBefore clipping:")
        print(f"  ||∇x|| = {grad_x_norm.item():.2f}")
        print(f"  ||∇W|| = {grad_W_norm.item():.2f}")

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_([x, W], max_norm)

        # Check after clipping
        grad_x_norm_clipped = torch.norm(x.grad)
        grad_W_norm_clipped = torch.norm(W.grad)

        print(f"\nAfter clipping (max_norm={max_norm}):")
        print(f"  ||∇x|| = {grad_x_norm_clipped.item():.2f}")
        print(f"  ||∇W|| = {grad_W_norm_clipped.item():.2f}")

        # Total norm
        total_norm = torch.sqrt(grad_x_norm_clipped**2 + grad_W_norm_clipped**2)
        print(f"  Total norm: {total_norm.item():.2f}")

    @staticmethod
    def mixed_precision_training():
        """
        Mixed precision training (FP16 + FP32)
        """
        print("\n" + "="*80)
        print("MIXED PRECISION TRAINING")
        print("="*80)

        print("""
Idea: Use FP16 for most computations, FP32 for critical parts

Benefits:
  ✅ 2x faster training (less memory bandwidth)
  ✅ 2x less GPU memory (fit larger models/batches)
  ✅ Same accuracy (with proper techniques)

Challenges:
  ❌ FP16 range: ~6 × 10^-5 to 6.5 × 10^4 (much smaller than FP32)
  ❌ Gradients often < 10^-5 → underflow!

Solutions:

1. Loss Scaling
   • Scale loss by large factor (e.g., 1024)
   • Prevents gradient underflow
   • Unscale before optimizer step

2. Master Weights (FP32)
   • Keep FP32 copy of weights
   • Accumulate small updates correctly

3. Dynamic Loss Scaling
   • Adjust scale factor automatically
   • Increase if no overflow, decrease if overflow

PyTorch: Use torch.cuda.amp (Automatic Mixed Precision)

Speedup: 1.5-3x typical
Memory savings: 2x
        """)

        # Example (pseudo-code, requires CUDA)
        print("\nPyTorch AMP example:")
        print("""
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    # Autocasts to FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Scales loss, backward in FP16
    scaler.scale(loss).backward()

    # Unscales gradients, clips, updates in FP32
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
        """)


# Demo
if __name__ == "__main__":
    stability = NumericalStability()

    stability.overflow_underflow_demo()
    stability.logsumexp_trick()
    stability.gradient_clipping()
    stability.mixed_precision_training()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - NUMERICAL STABILITY")
    print("="*80)
    print("""
1. FP32 has limited range and precision
   → Overflow (>10^38), underflow (<10^-38)

2. Log-sum-exp trick prevents overflow
   → Used in softmax, log-probabilities

3. Gradient clipping prevents explosion
   → Essential for LLM training
   → Typical max_norm: 1.0

4. Mixed precision (FP16/FP32) speeds training
   → 2x faster, 2x less memory
   → Use PyTorch AMP

5. Always monitor:
   → Loss (should decrease, not NaN/inf)
   → Gradient norms (check for explosion/vanishing)
   → Activations (check ranges)
    """)
```

## 6. Projet Complet: Training Loop avec Toutes les Math

```python
"""
PROJET: Complete Training Loop

Intègre tous les concepts mathématiques:
  • Linear algebra (matrix mult, dot products)
  • Calculus (backpropagation)
  • Probability (softmax, cross-entropy)
  • Optimization (Adam, LR schedule)
  • Numerical stability (gradient clipping, mixed precision)

Task: Train small Transformer on toy language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model
    vocab_size: int = 1000
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 1024

    # Training
    batch_size: int = 32
    seq_len: int = 64
    num_epochs: int = 10
    max_steps: int = 1000

    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 100

    # Numerical stability
    max_grad_norm: float = 1.0
    use_amp: bool = False

    # Logging
    log_interval: int = 10


class SimpleTransformerLM(nn.Module):
    """
    Simplified Transformer for language modeling

    Uses all math concepts from this chapter
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.config = config

        # Embeddings (Linear algebra: lookup table)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        # (Includes multi-head attention: Q·K^T, matrix multiplications)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Output projection
        self.output = nn.Linear(config.d_model, config.vocab_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights (Linear algebra: matrix initialization)"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embedding (Linear algebra: lookup)
        x = self.token_embedding(input_ids)

        # Scale (Numerical stability)
        x = x * math.sqrt(self.config.d_model)

        # Transformer (Matrix multiplications, dot products)
        x = self.transformer(x, mask=attention_mask)

        # Output projection (Linear algebra)
        logits = self.output(x)

        return logits


class ToyDataset(Dataset):
    """Toy dataset for demo"""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return input_ids, labels


class Trainer:
    """
    Complete training loop

    Demonstrates all math concepts
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config

        # Optimizer (Optimization: Adam)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Loss function (Probability: cross-entropy)
        self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler (Optimization: warmup + cosine)
        self.scheduler = self._create_scheduler()

        # Mixed precision (Numerical stability)
        if config.use_amp:
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Metrics
        self.step = 0
        self.epoch = 0

    def _create_scheduler(self):
        """Create LR scheduler with warmup"""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps

            # Cosine decay
            progress = (step - self.config.warmup_steps) / \
                      (self.config.max_steps - self.config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Single training step

        Demonstrates:
          • Forward pass (Linear algebra, matrix mult)
          • Loss computation (Probability: cross-entropy)
          • Backward pass (Calculus: backpropagation)
          • Gradient clipping (Numerical stability)
          • Optimizer step (Optimization: Adam)
        """
        self.model.train()

        # Forward pass (Linear algebra: matrix multiplications)
        logits = self.model(input_ids)

        # Compute loss (Probability: softmax + cross-entropy)
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)

        loss = self.criterion(logits, labels)

        # Backward pass (Calculus: compute gradients via chain rule)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (Numerical stability)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # Optimizer step (Optimization: Adam update rule)
        self.optimizer.step()

        # LR scheduler (Optimization: warmup + decay)
        self.scheduler.step()

        self.step += 1

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0

        start_time = time.time()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            loss = self.train_step(input_ids, labels)

            total_loss += loss
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                perplexity = math.exp(avg_loss)
                lr = self.scheduler.get_last_lr()[0]

                elapsed = time.time() - start_time
                print(f"  Step {self.step:4d} | "
                      f"Loss: {loss:.4f} | "
                      f"PPL: {perplexity:.2f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {elapsed:.1f}s")

            if self.step >= self.config.max_steps:
                break

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set"""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for input_ids, labels in dataloader:
                logits = self.model(input_ids)

                batch_size, seq_len, vocab_size = logits.shape
                logits = logits.view(-1, vocab_size)
                labels = labels.view(-1)

                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss


# Main / Demo
if __name__ == "__main__":
    print("="*80)
    print("COMPLETE TRAINING LOOP - ALL MATH CONCEPTS")
    print("="*80)

    # Configuration
    config = TrainingConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        batch_size=32,
        seq_len=64,
        num_epochs=3,
        max_steps=100,
        learning_rate=5e-4,
        warmup_steps=20,
        max_grad_norm=1.0
    )

    print("\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Seq length: {config.seq_len}")

    # Create model
    model = SimpleTransformerLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # Create datasets
    train_dataset = ToyDataset(1000, config.seq_len, config.vocab_size)
    val_dataset = ToyDataset(100, config.seq_len, config.vocab_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size
    )

    # Create trainer
    trainer = Trainer(model, config)

    # Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Evaluate
        val_loss = trainer.evaluate(val_loader)

        # Metrics
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        print(f"\n  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if trainer.step >= config.max_steps:
            break

    print("\n" + "="*80)
    print("✅ CHAPITRE 4 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant:

1. Linear Algebra
   ✅ Vecteurs, matrices, tensors
   ✅ Dot product, matrix multiplication
   ✅ Broadcasting

2. Calculus
   ✅ Derivatives, gradients
   ✅ Chain rule
   ✅ Backpropagation

3. Probability & Statistics
   ✅ Softmax
   ✅ Cross-entropy
   ✅ Perplexity
   ✅ Sampling strategies

4. Optimization
   ✅ SGD, Adam
   ✅ Learning rate schedules
   ✅ Warmup + decay

5. Numerical Stability
   ✅ Log-sum-exp trick
   ✅ Gradient clipping
   ✅ Mixed precision

Projet complet démontrant TOUS les concepts!

Prochaine étape:
  → Chapitre 5: Setup et Environnement de Développement
    (PyTorch, HuggingFace, CUDA, etc.)
    """)
```

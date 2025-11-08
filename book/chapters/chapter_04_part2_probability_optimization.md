# Chapitre 4 (Partie 2): Probability, Optimization et Numerical Stability

## 3. Probability & Statistics

```python
"""
Probability = Langage des LLMs

LLMs sont des modèles probabilistes:
  - Prédisent distribution de probabilité sur tokens
  - Sampling: Génèrent tokens selon probabilités
  - Training: Maximisent likelihood des données

Concepts clés:
  - Probability distributions
  - Softmax (convert logits → probabilities)
  - Cross-entropy loss
  - Perplexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ProbabilityForLLMs:
    """
    Probability concepts for language models
    """

    @staticmethod
    def softmax_explained():
        """
        Softmax = Convert scores to probabilities
        """
        print("="*80)
        print("SOFTMAX FUNCTION")
        print("="*80)

        print("""
Softmax: Convert logits (unbounded scores) → probabilities (0 to 1, sum to 1)

Formula:
  softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

Properties:
  1. Output in [0, 1]
  2. Sum to 1: Σ softmax(x_i) = 1
  3. Preserves order: if x_i > x_j, then softmax(x_i) > softmax(x_j)
  4. Differentiable (crucial for training!)

Example:
  Logits: [2.0, 1.0, 0.1]

  exp: [7.39, 2.72, 1.11]
  sum: 11.22

  Softmax: [0.659, 0.242, 0.099]
  → Sum = 1.0 ✓

In LLMs:
  • Attention: softmax(Q·K^T / √d_k)
    → Convert attention scores to weights

  • Output: softmax(logits)
    → Convert final scores to token probabilities
        """)

        # Example
        logits = torch.tensor([2.0, 1.0, 0.1])

        # Compute softmax manually
        exp_logits = torch.exp(logits)
        softmax_manual = exp_logits / exp_logits.sum()

        # PyTorch softmax
        softmax_torch = F.softmax(logits, dim=0)

        print("\nPyTorch example:")
        print(f"  Logits: {logits.tolist()}")
        print(f"  exp(logits): {exp_logits.tolist()}")
        print(f"  Softmax (manual): {softmax_manual.tolist()}")
        print(f"  Softmax (PyTorch): {softmax_torch.tolist()}")
        print(f"  Sum: {softmax_torch.sum().item():.6f}")

        # Temperature sampling
        print("\n--- Temperature Scaling ---")
        print("""
Temperature T controls randomness:
  softmax(x / T)

  • T = 1: Standard softmax
  • T → 0: Argmax (deterministic, picks highest)
  • T → ∞: Uniform (completely random)

T < 1: "Sharper" distribution (more confident)
T > 1: "Flatter" distribution (more random)
        """)

        temps = [0.5, 1.0, 2.0]
        for temp in temps:
            probs = F.softmax(logits / temp, dim=0)
            print(f"  T={temp}: {probs.tolist()}")

    @staticmethod
    def cross_entropy_explained():
        """
        Cross-entropy = Standard loss for classification
        """
        print("\n" + "="*80)
        print("CROSS-ENTROPY LOSS")
        print("="*80)

        print("""
Cross-Entropy: Measures difference between two probability distributions

Formula:
  H(p, q) = -Σ p(x) × log(q(x))

  where:
    p = true distribution (ground truth)
    q = predicted distribution (model output)

For classification (one-hot ground truth):
  Loss = -log(q[correct_class])

Properties:
  • Loss = 0 when q = p (perfect prediction)
  • Loss → ∞ when q[correct] → 0 (very wrong)
  • Convex (nice optimization landscape)

Example:
  True class: 2 (index)
  Predicted probs: [0.1, 0.2, 0.7]

  Loss = -log(0.7) = 0.357

In LLMs:
  • Language modeling: predict next token
  • True distribution: one-hot (correct token = 1, others = 0)
  • Predicted: softmax(logits)
  • Loss: Cross-entropy between true and predicted
        """)

        # Example
        # True class: 2
        target = torch.tensor([2])

        # Logits (before softmax)
        logits = torch.tensor([[2.0, 1.0, 3.0]])  # (batch=1, num_classes=3)

        # Compute loss
        loss = F.cross_entropy(logits, target)

        # Manual computation
        probs = F.softmax(logits, dim=1)
        manual_loss = -torch.log(probs[0, target[0]])

        print("\nPyTorch example:")
        print(f"  Logits: {logits[0].tolist()}")
        print(f"  Probabilities: {probs[0].tolist()}")
        print(f"  Target class: {target.item()}")
        print(f"  Loss (PyTorch): {loss.item():.4f}")
        print(f"  Loss (manual): {manual_loss.item():.4f}")

        # Relationship to perplexity
        print("\n--- Perplexity ---")
        print("""
Perplexity = exp(loss)

  Perplexity = exp(cross_entropy)

Interpretation:
  • Perplexity = effective vocabulary size the model is "confused" about
  • Lower is better
  • Perplexity of 50 → model as uncertain as if choosing uniformly from 50 tokens

Example:
  Loss = 3.0 → Perplexity = exp(3.0) ≈ 20
  Loss = 1.0 → Perplexity = exp(1.0) ≈ 2.7
  Loss = 0.1 → Perplexity = exp(0.1) ≈ 1.1
        """)

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Perplexity: {torch.exp(loss).item():.4f}")

    @staticmethod
    def sampling_strategies():
        """
        Sampling from probability distributions
        """
        print("\n" + "="*80)
        print("SAMPLING STRATEGIES")
        print("="*80)

        print("""
Given probabilities, how to generate next token?

1. Greedy Sampling
   → Always pick highest probability token
   → Deterministic, but repetitive

2. Random Sampling (Temperature)
   → Sample from softmax(logits / T)
   → More diverse, but can be incoherent

3. Top-k Sampling
   → Sample from top k tokens only
   → Filters unlikely tokens

4. Top-p (Nucleus) Sampling
   → Sample from smallest set with cumulative prob ≥ p
   → Dynamic vocabulary size

5. Beam Search
   → Keep track of k best sequences
   → Used for translation, summarization
        """)

        # Example logits
        logits = torch.tensor([3.0, 1.0, 0.5, 0.2, 0.1])
        vocab = ["the", "a", "is", "dog", "cat"]

        print(f"\nLogits: {dict(zip(vocab, logits.tolist()))}")

        # 1. Greedy
        greedy_idx = logits.argmax()
        print(f"\n1. Greedy: '{vocab[greedy_idx]}'")

        # 2. Temperature sampling
        temp = 0.8
        probs = F.softmax(logits / temp, dim=0)
        sampled_idx = torch.multinomial(probs, 1)
        print(f"2. Temperature (T={temp}): '{vocab[sampled_idx.item()]}'")

        # 3. Top-k sampling
        k = 3
        top_k_logits, top_k_indices = torch.topk(logits, k)
        top_k_probs = F.softmax(top_k_logits, dim=0)
        sampled_k = torch.multinomial(top_k_probs, 1)
        print(f"3. Top-k (k={k}): '{vocab[top_k_indices[sampled_k].item()]}'")

        # 4. Top-p (nucleus)
        p = 0.9
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=0)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Remove tokens with cumulative prob > p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Filter and sample
        filtered_logits = sorted_logits.clone()
        filtered_logits[sorted_indices_to_remove] = -float('inf')
        nucleus_probs = F.softmax(filtered_logits, dim=0)
        sampled_p = torch.multinomial(nucleus_probs, 1)

        print(f"4. Top-p (p={p}): '{vocab[sorted_indices[sampled_p].item()]}'")


# Demo
if __name__ == "__main__":
    prob = ProbabilityForLLMs()

    prob.softmax_explained()
    prob.cross_entropy_explained()
    prob.sampling_strategies()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - PROBABILITY")
    print("="*80)
    print("""
1. Softmax converts logits to probabilities
   → Σ probs = 1, all in [0,1]

2. Cross-entropy measures distribution difference
   → Standard loss for classification

3. Perplexity = exp(cross_entropy)
   → Interpretable metric (lower is better)

4. Sampling strategies control generation
   → Greedy: deterministic
   → Temperature: controls randomness
   → Top-k/Top-p: balance quality and diversity

5. Temperature is crucial for generation
   → T=0: deterministic
   → T=1: standard
   → T>1: more random
    """)
```

## 4. Optimization

```python
"""
Optimization = Comment améliorer le modèle

Goal: Minimize loss by adjusting weights

Gradient Descent:
  w = w - lr × ∇L

Challenges:
  - Learning rate selection
  - Local minima
  - Saddle points
  - Slow convergence

Solutions: Adam, AdamW, learning rate schedules
"""

class OptimizationForLLMs:
    """
    Optimization algorithms for training LLMs
    """

    @staticmethod
    def sgd_explained():
        """
        Stochastic Gradient Descent
        """
        print("="*80)
        print("STOCHASTIC GRADIENT DESCENT (SGD)")
        print("="*80)

        print("""
SGD: Update weights using gradient on mini-batch

Algorithm:
  1. Sample mini-batch of data
  2. Compute loss on mini-batch
  3. Compute gradients: ∇L
  4. Update weights: w = w - lr × ∇L

Why "Stochastic"?
  • Uses random mini-batch (not full dataset)
  • Faster than full-batch gradient descent
  • Adds noise → helps escape local minima

Learning Rate (lr):
  • Too small: slow convergence
  • Too large: divergence (loss explodes)
  • Typical: 1e-4 to 1e-1

With Momentum:
  v = β × v + ∇L
  w = w - lr × v

  → Accumulates velocity
  → Faster convergence
  → Smooths updates
        """)

        # Example
        torch.manual_seed(42)

        # Simple optimization problem: minimize (w - 3)²
        w = torch.tensor([0.0], requires_grad=True)
        target = 3.0

        optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)

        print("\nOptimizing w to reach target = 3.0")
        print("  Iteration | w      | Loss")
        print("  ----------|--------|--------")

        for i in range(10):
            optimizer.zero_grad()

            # Loss
            loss = (w - target) ** 2

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            if i % 2 == 0:
                print(f"  {i:9d} | {w.item():6.3f} | {loss.item():6.4f}")

        print(f"\n  Final w: {w.item():.6f} (target: {target})")

    @staticmethod
    def adam_explained():
        """
        Adam Optimizer
        """
        print("\n" + "="*80)
        print("ADAM OPTIMIZER")
        print("="*80)

        print("""
Adam: Adaptive Moment Estimation

Combines:
  • Momentum (first moment)
  • RMSProp (second moment, adaptive learning rates)

Algorithm:
  m = β₁ × m + (1 - β₁) × ∇L      (momentum)
  v = β₂ × v + (1 - β₂) × (∇L)²   (variance)

  m_hat = m / (1 - β₁^t)          (bias correction)
  v_hat = v / (1 - β₂^t)

  w = w - lr × m_hat / (√v_hat + ε)

Hyperparameters (typical):
  • lr = 1e-4 to 1e-3
  • β₁ = 0.9 (momentum)
  • β₂ = 0.999 (variance)
  • ε = 1e-8 (numerical stability)

Advantages:
  ✅ Adaptive learning rates per parameter
  ✅ Works well with sparse gradients
  ✅ Less sensitive to lr choice
  ✅ Standard for LLMs

AdamW (Weight Decay):
  Adam + L2 regularization (decoupled)
  → Better generalization
  → Used in GPT, BERT, etc.
        """)

        # Example: Compare SGD vs Adam
        torch.manual_seed(42)

        # Problem: minimize Rosenbrock function (hard optimization problem)
        def rosenbrock(x, y):
            return (1 - x)**2 + 100 * (y - x**2)**2

        # SGD
        x_sgd = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer_sgd = torch.optim.SGD([x_sgd], lr=0.001)

        # Adam
        x_adam = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer_adam = torch.optim.Adam([x_adam], lr=0.01)

        print("\nComparing SGD vs Adam (10 steps):")
        print("  Optimizing Rosenbrock function to (1, 1)")
        print("\n  Iter | SGD Loss | Adam Loss")
        print("  -----|----------|----------")

        for i in range(10):
            # SGD
            optimizer_sgd.zero_grad()
            loss_sgd = rosenbrock(x_sgd[0], x_sgd[1])
            loss_sgd.backward()
            optimizer_sgd.step()

            # Adam
            optimizer_adam.zero_grad()
            loss_adam = rosenbrock(x_adam[0], x_adam[1])
            loss_adam.backward()
            optimizer_adam.step()

            if i % 2 == 0:
                print(f"  {i:4d} | {loss_sgd.item():8.4f} | {loss_adam.item():9.4f}")

        print(f"\n  SGD final: x={x_sgd[0].item():.4f}, y={x_sgd[1].item():.4f}")
        print(f"  Adam final: x={x_adam[0].item():.4f}, y={x_adam[1].item():.4f}")
        print(f"  Target: x=1.0, y=1.0")

    @staticmethod
    def learning_rate_schedules():
        """
        Learning rate scheduling strategies
        """
        print("\n" + "="*80)
        print("LEARNING RATE SCHEDULES")
        print("="*80)

        print("""
Problem: Fixed LR is suboptimal
  • High LR early: fast progress but unstable
  • Low LR late: stable convergence

Solution: Schedule LR during training

Popular Schedules:

1. Linear Warmup + Decay
   - Warmup: Increase LR linearly for N steps
   - Decay: Decrease LR (linear, cosine, or exponential)
   - Used in: BERT, GPT

2. Cosine Annealing
   lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(πt/T))
   - Smooth decrease
   - Used in: Many modern LLMs

3. OneCycleLR
   - Increase to max_lr, then decrease
   - Fast convergence
   - Used in: Fast.ai

4. Noam (Transformer paper)
   lr(step) = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
   - Warmup + inverse sqrt decay
   - Used in: Original Transformer

Why Warmup?
  • Prevents large updates at start (unstable)
  • Adam's variance estimate unreliable early on
  • Typical warmup: 10-20% of training
        """)

        # Visualize schedules
        import matplotlib.pyplot as plt

        steps = 1000
        warmup_steps = 100
        d_model = 512

        # Noam schedule
        lr_noam = []
        for step in range(1, steps + 1):
            lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
            lr_noam.append(lr)

        # Cosine schedule with warmup
        max_lr = 1e-3
        min_lr = 1e-5
        lr_cosine = []

        for step in range(1, steps + 1):
            if step < warmup_steps:
                lr = max_lr * (step / warmup_steps)
            else:
                progress = (step - warmup_steps) / (steps - warmup_steps)
                lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
            lr_cosine.append(lr)

        print("\n  Example learning rate values:")
        print(f"  Step 1: Noam={lr_noam[0]:.6f}, Cosine={lr_cosine[0]:.6f}")
        print(f"  Step 100: Noam={lr_noam[99]:.6f}, Cosine={lr_cosine[99]:.6f}")
        print(f"  Step 500: Noam={lr_noam[499]:.6f}, Cosine={lr_cosine[499]:.6f}")
        print(f"  Step 1000: Noam={lr_noam[999]:.6f}, Cosine={lr_cosine[999]:.6f}")


# Demo
if __name__ == "__main__":
    optim = OptimizationForLLMs()

    optim.sgd_explained()
    optim.adam_explained()
    optim.learning_rate_schedules()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - OPTIMIZATION")
    print("="*80)
    print("""
1. SGD: Simple but effective
   → Add momentum for better convergence

2. Adam: Adaptive learning rates
   → Standard choice for LLMs
   → Use AdamW for weight decay

3. Learning rate is critical
   → Too high: divergence
   → Too low: slow convergence

4. Use LR schedules
   → Warmup + decay
   → Cosine annealing popular

5. Typical LR for LLMs: 1e-4 to 1e-3
   → Fine-tuning: lower (1e-5 to 1e-4)
   → Pre-training: higher (1e-4 to 1e-3)
    """)
```

*[Suite avec Numerical Stability et projet complet dans la partie 3...]*

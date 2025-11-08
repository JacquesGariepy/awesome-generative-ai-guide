# Chapitre 4: Mathématiques pour les LLMs

## Introduction

Les **mathématiques** sont au cœur des LLMs. Pas besoin d'un PhD en maths, mais comprendre les concepts clés vous permet de:
- Débugger vos modèles efficacement
- Optimiser les hyperparamètres intelligemment
- Comprendre les papers de recherche
- Implémenter des architectures from scratch

### Ce que vous allez apprendre

```python
"""
Mathématiques essentielles pour LLMs:

1. Linear Algebra (Algèbre Linéaire)
   - Vecteurs et matrices
   - Produit scalaire (dot product)
   - Multiplication matricielle
   - → Utilisé dans: Attention, embeddings, layers

2. Calculus (Calcul)
   - Dérivées et gradients
   - Chain rule (règle de la chaîne)
   - Backpropagation
   - → Utilisé dans: Training (optimisation)

3. Probability & Statistics
   - Distributions de probabilité
   - Softmax
   - Cross-entropy
   - → Utilisé dans: Loss functions, sampling

4. Optimization
   - Gradient Descent
   - Adam optimizer
   - Learning rate schedules
   - → Utilisé dans: Training loop

5. Numerical Stability
   - Log-sum-exp trick
   - Gradient clipping
   - Mixed precision
   - → Utilisé dans: Preventing NaN/Inf

Tous avec implémentations PyTorch!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math
```

## 1. Linear Algebra (Algèbre Linéaire)

```python
"""
Linear Algebra = Fondation de TOUT en deep learning

Concepts clés:
  - Scalar: Un nombre (0D)
  - Vector: Liste de nombres (1D)
  - Matrix: Tableau 2D de nombres
  - Tensor: Tableau N-D (généralisation)

Operations:
  - Dot product (produit scalaire)
  - Matrix multiplication
  - Transpose
  - Broadcasting
"""

class LinearAlgebraBasics:
    """
    Core linear algebra operations for LLMs
    """

    @staticmethod
    def demonstrate_shapes():
        """
        Understanding tensor shapes
        """
        print("="*80)
        print("TENSOR SHAPES IN LLMS")
        print("="*80)

        examples = [
            {
                "name": "Scalar",
                "shape": "()",
                "dims": "0D",
                "example": "loss = 2.5",
                "usage": "Loss values, hyperparameters"
            },
            {
                "name": "Vector",
                "shape": "(n,)",
                "dims": "1D",
                "example": "embedding = [0.1, -0.2, 0.5, ...]",
                "usage": "Single token embedding, biases"
            },
            {
                "name": "Matrix",
                "shape": "(m, n)",
                "dims": "2D",
                "example": "W = [[0.1, 0.2], [0.3, 0.4]]",
                "usage": "Weight matrices, attention scores"
            },
            {
                "name": "Batch of Vectors",
                "shape": "(batch, n)",
                "dims": "2D",
                "example": "embeddings for batch",
                "usage": "Batch of embeddings"
            },
            {
                "name": "Sequence",
                "shape": "(batch, seq_len, d_model)",
                "dims": "3D",
                "example": "Token embeddings",
                "usage": "Transformer inputs/outputs"
            },
            {
                "name": "Attention Weights",
                "shape": "(batch, num_heads, seq_len, seq_len)",
                "dims": "4D",
                "example": "Multi-head attention",
                "usage": "Attention patterns"
            }
        ]

        for ex in examples:
            print(f"\n{ex['name']}")
            print(f"  Shape: {ex['shape']}")
            print(f"  Dimensions: {ex['dims']}")
            print(f"  Example: {ex['example']}")
            print(f"  Usage: {ex['usage']}")

    @staticmethod
    def dot_product_explained():
        """
        Dot product = Core operation in attention
        """
        print("\n" + "="*80)
        print("DOT PRODUCT (Produit Scalaire)")
        print("="*80)

        print("""
Formula:
  a · b = Σ(a_i × b_i) = a_1×b_1 + a_2×b_2 + ... + a_n×b_n

Example:
  a = [1, 2, 3]
  b = [4, 5, 6]
  a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

Geometric Interpretation:
  a · b = ||a|| × ||b|| × cos(θ)
  where θ = angle between vectors

  → If θ = 0° (same direction): cos(0) = 1 → maximum dot product
  → If θ = 90° (orthogonal): cos(90°) = 0 → dot product = 0
  → If θ = 180° (opposite): cos(180°) = -1 → negative dot product

Importance in LLMs:
  • Attention: Q·K^T computes similarity between queries and keys
  • High dot product → High similarity → High attention weight
        """)

        # Example
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        dot = torch.dot(a, b)

        print("\nPyTorch example:")
        print(f"  a = {a}")
        print(f"  b = {b}")
        print(f"  a · b = {dot.item()}")

        # Similarity example
        vec1 = torch.tensor([1.0, 0.0])  # Points right
        vec2 = torch.tensor([1.0, 0.0])  # Same direction
        vec3 = torch.tensor([0.0, 1.0])  # Orthogonal (points up)
        vec4 = torch.tensor([-1.0, 0.0])  # Opposite direction

        print("\n  Similarity examples:")
        print(f"  [1,0] · [1,0] = {torch.dot(vec1, vec2).item():.1f} (same direction)")
        print(f"  [1,0] · [0,1] = {torch.dot(vec1, vec3).item():.1f} (orthogonal)")
        print(f"  [1,0] · [-1,0] = {torch.dot(vec1, vec4).item():.1f} (opposite)")

    @staticmethod
    def matrix_multiplication_explained():
        """
        Matrix multiplication = Building block of neural networks
        """
        print("\n" + "="*80)
        print("MATRIX MULTIPLICATION")
        print("="*80)

        print("""
Rule: (m, n) × (n, p) → (m, p)
  - Inner dimensions (n) must match
  - Result has outer dimensions (m, p)

Example:
  A (2, 3) × B (3, 4) → C (2, 4) ✅
  A (2, 3) × B (5, 4) → ERROR! ❌ (3 ≠ 5)

Computation:
  C[i,j] = Σ_k A[i,k] × B[k,j]
  (Row i of A · Column j of B)

In LLMs:
  • Linear layers: y = x·W + b
    * x: (batch, d_in)
    * W: (d_in, d_out)
    * y: (batch, d_out)

  • Attention: Q·K^T
    * Q: (batch, seq_len, d_k)
    * K^T: (batch, d_k, seq_len)
    * Scores: (batch, seq_len, seq_len)
        """)

        # Example
        A = torch.tensor([
            [1, 2],
            [3, 4]
        ], dtype=torch.float32)

        B = torch.tensor([
            [5, 6],
            [7, 8]
        ], dtype=torch.float32)

        C = torch.matmul(A, B)

        print("\nPyTorch example:")
        print(f"  A ({A.shape}):")
        print(f"    {A[0].tolist()}")
        print(f"    {A[1].tolist()}")
        print(f"\n  B ({B.shape}):")
        print(f"    {B[0].tolist()}")
        print(f"    {B[1].tolist()}")
        print(f"\n  A @ B ({C.shape}):")
        print(f"    {C[0].tolist()}")
        print(f"    {C[1].tolist()}")

        # Verify computation
        print(f"\n  Verification C[0,0]:")
        print(f"    = A[0,:] · B[:,0]")
        print(f"    = [1,2] · [5,7]")
        print(f"    = 1×5 + 2×7 = {1*5 + 2*7}")

    @staticmethod
    def broadcasting_explained():
        """
        Broadcasting = Implicit expansion for element-wise ops
        """
        print("\n" + "="*80)
        print("BROADCASTING")
        print("="*80)

        print("""
Broadcasting allows operations on tensors of different shapes.

Rules:
  1. If ranks differ, prepend 1s to smaller shape
  2. Dimensions are compatible if:
     - They are equal, OR
     - One of them is 1

Examples:
  (3, 1) + (1, 4) → (3, 4) ✅  (broadcast both)
  (5, 3, 4) + (3, 4) → (5, 3, 4) ✅  (broadcast second)
  (3, 4) + (5, 1) → ERROR ❌  (3 ≠ 5 and neither is 1)

In LLMs:
  • Adding bias: (batch, seq_len, d_model) + (d_model,)
  • Scaling: embeddings * √d_model
  • Masking: scores + mask
        """)

        # Example 1: Add bias
        x = torch.randn(2, 3, 4)  # (batch, seq_len, d_model)
        bias = torch.randn(4)     # (d_model,)

        y = x + bias  # Broadcasts to (2, 3, 4)

        print("\nExample 1: Adding bias")
        print(f"  x: {x.shape}")
        print(f"  bias: {bias.shape}")
        print(f"  x + bias: {y.shape}")

        # Example 2: Masking
        scores = torch.randn(2, 5, 5)  # (batch, seq_len, seq_len)
        mask = torch.tril(torch.ones(5, 5))  # (seq_len, seq_len)

        masked_scores = scores + (mask - 1) * 1e9

        print("\nExample 2: Applying causal mask")
        print(f"  scores: {scores.shape}")
        print(f"  mask: {mask.shape}")
        print(f"  masked_scores: {masked_scores.shape}")


# Demo
if __name__ == "__main__":
    algebra = LinearAlgebraBasics()

    algebra.demonstrate_shapes()
    algebra.dot_product_explained()
    algebra.matrix_multiplication_explained()
    algebra.broadcasting_explained()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - LINEAR ALGEBRA")
    print("="*80)
    print("""
1. Tensors are N-dimensional arrays
   → Scalars (0D), Vectors (1D), Matrices (2D), etc.

2. Dot Product measures similarity
   → Used in attention (Q·K^T)

3. Matrix Multiplication builds layers
   → Linear layers: y = x·W + b

4. Broadcasting enables flexible operations
   → Add bias, apply masks, scale values

5. Shape compatibility is critical
   → Always check tensor shapes!
   → Use .shape to debug

Practice:
  - Implement attention from scratch (Chapter 2)
  - Verify shapes at each step
  - Use einsum for complex operations
    """)
```

## 2. Calculus (Calcul) et Backpropagation

```python
"""
Calculus = Comment les modèles APPRENNENT

Concepts:
  - Derivative: Rate of change
  - Gradient: Vector of partial derivatives
  - Chain Rule: Compose derivatives through network
  - Backpropagation: Efficiently compute gradients

Goal: Adjust weights to minimize loss
"""

class CalculusForLLMs:
    """
    Calculus concepts for neural network training
    """

    @staticmethod
    def derivative_basics():
        """
        Explain derivatives and gradients
        """
        print("="*80)
        print("DERIVATIVES AND GRADIENTS")
        print("="*80)

        print("""
Derivative:
  Measures how much output changes when input changes slightly

  f'(x) = lim(h→0) [f(x+h) - f(x)] / h

Example:
  f(x) = x²
  f'(x) = 2x

  At x=3: f'(3) = 6
  → If we increase x by 0.01, f increases by ~0.06

Gradient:
  Vector of partial derivatives (multi-variable function)

  f(x, y) = x² + y²
  ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

In Neural Networks:
  • Loss L(w₁, w₂, ..., wₙ)
  • Gradient: ∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
  • Tells us direction to adjust weights to reduce loss

PyTorch computes gradients automatically!
        """)

        # Example: Compute gradient
        x = torch.tensor([3.0], requires_grad=True)
        y = x ** 2

        # Backward pass
        y.backward()

        print("\nPyTorch example:")
        print(f"  f(x) = x²")
        print(f"  x = {x.item()}")
        print(f"  f(x) = {y.item()}")
        print(f"  f'(x) = {x.grad.item()} (computed by PyTorch)")
        print(f"  Expected: 2x = {2 * x.item()}")

    @staticmethod
    def chain_rule_explained():
        """
        Chain rule = Foundation of backpropagation
        """
        print("\n" + "="*80)
        print("CHAIN RULE")
        print("="*80)

        print("""
Chain Rule: Compose derivatives through functions

Formula:
  If y = g(f(x)), then:
  dy/dx = (dy/dg) × (dg/dx)

Example:
  f(x) = x²
  g(u) = u + 1
  y = g(f(x)) = x² + 1

  dy/dx = (dy/dg) × (dg/df) × (df/dx)
        = 1 × 1 × 2x
        = 2x

In Neural Networks (3 layers):
  x → Linear → ReLU → Linear → Loss

  ∂Loss/∂W₁ = (∂Loss/∂Linear₂) × (∂Linear₂/∂ReLU) ×
               (∂ReLU/∂Linear₁) × (∂Linear₁/∂W₁)

  → Backpropagation applies chain rule backwards!
        """)

        # Example
        x = torch.tensor([2.0], requires_grad=True)

        # Forward: y = (x²)² = x⁴
        u = x ** 2      # u = x²
        y = u ** 2      # y = u²

        # Backward
        y.backward()

        print("\nPyTorch example:")
        print(f"  y = (x²)² = x⁴")
        print(f"  x = {x.item()}")
        print(f"  dy/dx = {x.grad.item()} (PyTorch)")
        print(f"  Expected: 4x³ = {4 * x.item()**3}")

    @staticmethod
    def backpropagation_walkthrough():
        """
        Complete backpropagation example
        """
        print("\n" + "="*80)
        print("BACKPROPAGATION WALKTHROUGH")
        print("="*80)

        print("""
Simple network: x → W₁ → ReLU → W₂ → Loss

Forward:
  1. h = x·W₁
  2. a = ReLU(h) = max(0, h)
  3. y = a·W₂
  4. L = (y - target)²

Backward (compute ∂L/∂W₁, ∂L/∂W₂):
  1. ∂L/∂y = 2(y - target)
  2. ∂L/∂W₂ = (∂L/∂y) × (∂y/∂W₂) = ∂L/∂y × a
  3. ∂L/∂a = (∂L/∂y) × (∂y/∂a) = ∂L/∂y × W₂
  4. ∂L/∂h = (∂L/∂a) × (∂a/∂h) = ∂L/∂a × (h > 0)
  5. ∂L/∂W₁ = (∂L/∂h) × (∂h/∂W₁) = ∂L/∂h × x

Then update:
  W₁ = W₁ - lr × ∂L/∂W₁
  W₂ = W₂ - lr × ∂L/∂W₂
        """)

        # Implement from scratch
        torch.manual_seed(42)

        # Data
        x = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[5.0]])

        # Weights
        W1 = torch.randn(2, 3, requires_grad=True)
        W2 = torch.randn(3, 1, requires_grad=True)

        # Forward
        h = x @ W1
        a = F.relu(h)
        y = a @ W2
        loss = ((y - target) ** 2).mean()

        print(f"\nForward pass:")
        print(f"  x: {x.shape}")
        print(f"  h = x @ W1: {h.shape}")
        print(f"  a = ReLU(h): {a.shape}")
        print(f"  y = a @ W2: {y.shape}")
        print(f"  loss = (y - target)²: {loss.item():.4f}")

        # Backward (automatic)
        loss.backward()

        print(f"\nBackward pass (gradients):")
        print(f"  ∂L/∂W1: {W1.grad.shape}")
        print(f"  ∂L/∂W2: {W2.grad.shape}")

        # Update
        lr = 0.01
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad

        print(f"\nWeights updated with lr={lr}")


# Demo
if __name__ == "__main__":
    calculus = CalculusForLLMs()

    calculus.derivative_basics()
    calculus.chain_rule_explained()
    calculus.backpropagation_walkthrough()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - CALCULUS")
    print("="*80)
    print("""
1. Derivative = Rate of change
   → How much output changes with input

2. Gradient = Vector of partial derivatives
   → Direction of steepest increase

3. Chain Rule = Compose derivatives
   → Foundation of backpropagation

4. Backpropagation = Efficient gradient computation
   → Applies chain rule backwards through network

5. PyTorch autograd does this automatically!
   → Call .backward() and gradients are computed

Practice:
  - Implement simple MLP from scratch
  - Verify gradients with finite differences
  - Understand .backward() step-by-step
    """)
```

*[Suite avec Probability, Optimization et Numerical Stability dans la partie 2...]*

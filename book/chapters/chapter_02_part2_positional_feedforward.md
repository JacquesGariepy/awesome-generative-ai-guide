# Chapitre 2 (Partie 2): Positional Encoding et Feed-Forward Networks

## 3. Positional Encoding

```python
"""
Problème: Self-Attention n'a AUCUNE notion de position!

Exemple:
  "The cat sat on the mat"
  "mat the on sat cat The"

→ Pour self-attention, ces deux phrases sont IDENTIQUES!
  (même ensemble de tokens, ordre différent ne change rien)

Solution: Positional Encoding

Ajouter information de position à chaque token embedding:
  embedding_final = token_embedding + positional_encoding

Deux approches:

1. Sinusoidal (Vaswani et al., 2017)
   - Formule mathématique fixe
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   - Avantage: Peut extrapoler à séquences plus longues
   - Utilisé dans: Transformer original

2. Learned (appris pendant training)
   - Embedding table pour chaque position
   - Avantage: Plus flexible, apprend patterns spécifiques
   - Inconvénient: Taille fixe (max_seq_len)
   - Utilisé dans: GPT, BERT
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (from original Transformer)

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where:
        pos = position in sequence (0, 1, 2, ...)
        i = dimension index (0, 1, 2, ..., d_model/2)

    Properties:
        - Unique encoding for each position
        - Can extrapolate to longer sequences
        - Relative positions: PE(pos+k) is linear function of PE(pos)

    Args:
        d_model: Model dimension (must be even)
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute div_term for all dimensions
        # div_term = 10000^(2i/d_model) for i in [0, d_model/2)
        # Equivalent to: exp(-log(10000) * 2i / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)

        # Add positional encoding (broadcast over batch)
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (used in BERT, GPT)

    Simple embedding table for positions.
    More flexible but limited to max_len.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Embedding table for positions
        self.position_embeddings = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Register position indices as buffer
        self.register_buffer(
            'position_ids',
            torch.arange(max_len).expand((1, -1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            x + learned positional encoding
        """
        seq_len = x.size(1)

        # Get position embeddings for sequence
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)

        # Add to input
        x = x + position_embeddings

        return self.dropout(x)


class PositionalEncodingVisualizer:
    """
    Visualize and analyze positional encodings
    """

    @staticmethod
    def visualize_sinusoidal(d_model: int = 128, max_len: int = 100):
        """
        Visualize sinusoidal positional encoding

        Creates heatmap showing encoding patterns
        """
        # Create encoding
        pe_module = SinusoidalPositionalEncoding(d_model, max_len)
        pe = pe_module.pe.squeeze(0).numpy()  # (max_len, d_model)

        print("="*80)
        print("SINUSOIDAL POSITIONAL ENCODING VISUALIZATION")
        print("="*80)
        print(f"\nShape: {pe.shape}")
        print(f"  Positions: {max_len}")
        print(f"  Dimensions: {d_model}")

        # Statistics
        print(f"\nStatistics:")
        print(f"  Min: {pe.min():.4f}")
        print(f"  Max: {pe.max():.4f}")
        print(f"  Mean: {pe.mean():.4f}")
        print(f"  Std: {pe.std():.4f}")

        # Show pattern for first few positions
        print(f"\nFirst 5 positions (first 8 dimensions):")
        for pos in range(5):
            values_str = " ".join(f"{v:6.3f}" for v in pe[pos, :8])
            print(f"  pos {pos}: [{values_str}]")

    @staticmethod
    def compare_encodings():
        """
        Compare sinusoidal vs learned encodings
        """
        print("\n" + "="*80)
        print("SINUSOIDAL VS LEARNED POSITIONAL ENCODING")
        print("="*80)

        comparison = [
            {
                "aspect": "Type",
                "sinusoidal": "Fixed mathematical formula",
                "learned": "Learned during training"
            },
            {
                "aspect": "Parameters",
                "sinusoidal": "0 (no trainable params)",
                "learned": "max_len × d_model"
            },
            {
                "aspect": "Extrapolation",
                "sinusoidal": "✅ Can handle longer sequences",
                "learned": "❌ Limited to max_len"
            },
            {
                "aspect": "Flexibility",
                "sinusoidal": "Fixed pattern",
                "learned": "Learns task-specific patterns"
            },
            {
                "aspect": "Performance",
                "sinusoidal": "Good for general tasks",
                "learned": "Often slightly better"
            },
            {
                "aspect": "Used in",
                "sinusoidal": "Original Transformer, some models",
                "learned": "BERT, GPT, most modern LLMs"
            }
        ]

        for item in comparison:
            print(f"\n{item['aspect']}:")
            print(f"  Sinusoidal: {item['sinusoidal']}")
            print(f"  Learned: {item['learned']}")

    @staticmethod
    def demonstrate_position_sensitivity():
        """
        Show that positional encoding makes model position-aware
        """
        print("\n" + "="*80)
        print("POSITION SENSITIVITY DEMONSTRATION")
        print("="*80)

        d_model = 64
        max_len = 10

        # Same token at different positions
        token_embedding = torch.randn(1, 1, d_model)

        pe_module = SinusoidalPositionalEncoding(d_model, max_len)

        print("\nSame token embedding at different positions:")
        print(f"Token embedding shape: {token_embedding.shape}")

        # Position 0
        pos_0 = torch.zeros(1, 1, d_model)
        pos_0 = pos_0 + pe_module.pe[:, 0:1, :]
        output_0 = token_embedding + pos_0

        # Position 5
        pos_5 = torch.zeros(1, 1, d_model)
        pos_5 = pos_5 + pe_module.pe[:, 5:6, :]
        output_5 = token_embedding + pos_5

        # Compute difference
        diff = torch.norm(output_0 - output_5).item()

        print(f"\n  Position 0 output (first 8 dims): {output_0[0, 0, :8].numpy().round(3)}")
        print(f"  Position 5 output (first 8 dims): {output_5[0, 0, :8].numpy().round(3)}")
        print(f"\n  L2 distance: {diff:.4f}")
        print("\n  ✅ Same token, different positions → Different representations!")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("POSITIONAL ENCODING")
    print("="*80)

    # Configuration
    d_model = 512
    max_len = 100
    seq_len = 20
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  max_len: {max_len}")
    print(f"  seq_len: {seq_len}")

    # Test sinusoidal
    print("\n--- Sinusoidal Positional Encoding ---")
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_len)

    x = torch.randn(batch_size, seq_len, d_model)
    output = sinusoidal_pe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in sinusoidal_pe.parameters() if p.requires_grad)}")

    # Test learned
    print("\n--- Learned Positional Encoding ---")
    learned_pe = LearnedPositionalEncoding(d_model, max_len)

    output = learned_pe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in learned_pe.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"  → Embedding table: {max_len} × {d_model} = {max_len * d_model:,}")

    # Visualizations
    visualizer = PositionalEncodingVisualizer()
    visualizer.visualize_sinusoidal(d_model=128, max_len=100)
    visualizer.compare_encodings()
    visualizer.demonstrate_position_sensitivity()
```

## 4. Position-wise Feed-Forward Networks

```python
"""
Feed-Forward Network (FFN) = MLP appliqué à chaque position

Architecture simple:
  FFN(x) = max(0, x·W1 + b1)·W2 + b2
         = ReLU(Linear(x))·Linear

Dimensions (typique):
  - Input: d_model (e.g., 512, 768)
  - Hidden: d_ff = 4 × d_model (e.g., 2048, 3072)
  - Output: d_model

Pourquoi 4x expansion?
  - Plus de capacity pour apprendre transformations complexes
  - Permet au modèle de "penser" avant de projeter vers sortie
  - Empiriquement prouvé optimal

Position-wise = Appliqué indépendamment à chaque position
  - Même FFN pour tous les tokens
  - Pas d'interaction entre positions (contrairement à attention)

Role dans Transformer:
  - Attention: Mixing information across positions
  - FFN: Processing information at each position
"""

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    Two linear transformations with ReLU/GELU activation.
    Applied identically to each position.

    FFN(x) = activation(x·W1 + b1)·W2 + b2

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension (typically 4 × d_model)
        dropout: Dropout rate
        activation: Activation function ('relu' or 'gelu')
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # First linear layer: d_model → d_ff (expansion)
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear layer: d_ff → d_model (projection)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            Output (batch, seq_len, d_model)
        """
        # Expand: (batch, seq_len, d_model) → (batch, seq_len, d_ff)
        x = self.linear1(x)

        # Activation
        x = self.activation(x)

        # Dropout
        x = self.dropout(x)

        # Project back: (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        x = self.linear2(x)

        # Dropout
        x = self.dropout(x)

        return x


class FFNVariants:
    """
    Variants of Feed-Forward Networks in modern LLMs
    """

    @staticmethod
    def explain_variants():
        """
        Different FFN architectures used in various models
        """
        print("="*80)
        print("FEED-FORWARD NETWORK VARIANTS")
        print("="*80)

        variants = [
            {
                "name": "Standard FFN (Transformer)",
                "formula": "FFN(x) = ReLU(x·W1)·W2",
                "d_ff": "4 × d_model",
                "activation": "ReLU",
                "used_in": ["Original Transformer", "BERT"],
                "params": "2 × d_model × d_ff"
            },

            {
                "name": "GELU FFN (GPT)",
                "formula": "FFN(x) = GELU(x·W1)·W2",
                "d_ff": "4 × d_model",
                "activation": "GELU (smoother than ReLU)",
                "used_in": ["GPT-2", "GPT-3", "BERT (later versions)"],
                "params": "2 × d_model × d_ff"
            },

            {
                "name": "SwiGLU (Llama, PaLM)",
                "formula": "FFN(x) = (Swish(x·W1) ⊙ x·V)·W2",
                "d_ff": "8/3 × d_model (compensate for gating)",
                "activation": "Swish + Gating",
                "used_in": ["Llama 2", "PaLM", "Mixtral"],
                "params": "3 × d_model × d_ff (gate adds params)"
            },

            {
                "name": "GLU Variants",
                "formula": "FFN(x) = (σ(x·W1) ⊙ x·V)·W2",
                "d_ff": "Variable",
                "activation": "Gated Linear Unit",
                "used_in": ["Various research models"],
                "params": "3 × d_model × d_ff"
            }
        ]

        for var in variants:
            print(f"\n{var['name']}")
            print(f"  Formula: {var['formula']}")
            print(f"  Hidden size: {var['d_ff']}")
            print(f"  Activation: {var['activation']}")
            print(f"  Used in: {', '.join(var['used_in'])}")
            print(f"  Parameters: {var['params']}")

        print("\n" + "="*80)
        print("WHY 4x EXPANSION?")
        print("="*80)
        print("""
1. Increased Capacity
   - More parameters → Can learn more complex transformations
   - Empirically optimal (from Vaswani et al., 2017)

2. Information Processing
   - Attention: Mixes information across positions
   - FFN: Processes information at each position
   - Needs capacity to transform representations

3. Ablation Studies
   - 2x: Underperforms
   - 4x: Sweet spot (performance vs cost)
   - 8x: Marginal gains, much higher cost

4. Modern Trends
   - Some models use different ratios
   - Llama 2: ~8/3x (for SwiGLU)
   - Trade-off between params and performance
        """)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in Llama 2, PaLM)

    SwiGLU(x) = (Swish(x·W1) ⊙ x·V)·W2

    where:
        Swish(x) = x · sigmoid(βx)
        ⊙ = element-wise product (gating)

    More parameters but better performance.

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Three linear layers (instead of two)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            Output (batch, seq_len, d_model)
        """
        # Swish activation
        swish = F.silu(self.W1(x))  # SiLU = Swish

        # Gating
        gate = self.V(x)

        # Element-wise product
        x = swish * gate

        # Dropout
        x = self.dropout(x)

        # Project back
        x = self.W2(x)

        # Dropout
        x = self.dropout(x)

        return x


# Demo
if __name__ == "__main__":
    print("="*80)
    print("POSITION-WISE FEED-FORWARD NETWORK")
    print("="*80)

    # Configuration (GPT-2 small)
    d_model = 768
    d_ff = 4 * d_model  # 3072
    seq_len = 20
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff} (4x expansion)")

    # Standard FFN
    print("\n--- Standard FFN (GELU) ---")
    ffn = PositionwiseFeedForward(d_model, d_ff, activation='gelu')

    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"  Linear1 (W + b): {d_model * d_ff + d_ff:,}")
    print(f"  Linear2 (W + b): {d_ff * d_model + d_model:,}")

    # SwiGLU
    print("\n--- SwiGLU FFN (Llama 2 style) ---")
    # Llama uses ~8/3 expansion for SwiGLU
    d_ff_swiglu = int(8 * d_model / 3)
    swiglu = SwiGLU(d_model, d_ff_swiglu)

    output = swiglu(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in swiglu.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"  (3 linear layers due to gating)")

    # Explain variants
    FFNVariants.explain_variants()
```

## 5. Layer Normalization et Residual Connections

```python
"""
Layer Normalization + Residual Connections = Critical pour training profond

Problème: Deep networks sont difficiles à entraîner
  - Vanishing/exploding gradients
  - Internal covariate shift

Solutions:

1. Residual Connection (Skip Connection)
   output = Layer(x) + x

   Avantages:
   - Gradients flow directly à travers réseau
   - Permet d'entraîner réseaux très profonds (100+ layers)
   - Identity mapping: si Layer(x) = 0, output = x (no harm)

2. Layer Normalization
   Normalise activations pour chaque exemple indépendamment

   LN(x) = γ · (x - μ) / σ + β

   where:
   - μ = mean across features
   - σ = standard deviation
   - γ, β = learned parameters

Deux variantes dans Transformers:

A) Post-LN (Original Transformer)
   x = x + LN(Layer(x))

B) Pre-LN (Modern, plus stable)
   x = x + Layer(LN(x))

Pre-LN est maintenant standard (GPT, Llama, etc.)
"""

class LayerNorm(nn.Module):
    """
    Layer Normalization

    Normalizes activations across features for each example.

    Formula:
        LN(x) = γ · (x - μ) / (σ + ε) + β

    where:
        μ = mean(x) across last dimension
        σ = std(x) across last dimension
        γ, β = learnable parameters
        ε = small constant for numerical stability

    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            Normalized output
        """
        # Compute mean and std across last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Normalize
        x_norm = (x - mean) / (std + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta


class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization

    Implements: x + Dropout(Layer(LayerNorm(x)))
    (Pre-LN variant, more stable)

    Args:
        d_model: Model dimension
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: nn.Module
    ) -> torch.Tensor:
        """
        Apply residual connection

        Args:
            x: Input
            sublayer: The layer to apply (e.g., attention, FFN)

        Returns:
            x + sublayer(norm(x))
        """
        # Pre-LN: normalize first
        x_norm = self.norm(x)

        # Apply sublayer
        sublayer_output = sublayer(x_norm)

        # Dropout
        sublayer_output = self.dropout(sublayer_output)

        # Residual connection
        return x + sublayer_output


class NormalizationComparison:
    """
    Compare different normalization techniques
    """

    @staticmethod
    def explain_normalizations():
        """
        Compare Batch Norm, Layer Norm, RMS Norm
        """
        print("="*80)
        print("NORMALIZATION TECHNIQUES COMPARISON")
        print("="*80)

        norms = [
            {
                "name": "Batch Normalization",
                "normalize_over": "Batch dimension",
                "formula": "(x - μ_batch) / σ_batch",
                "used_in": "CNNs, Computer Vision",
                "pros": "Reduces internal covariate shift",
                "cons": "Batch-dependent, poor for small batches, not ideal for sequences"
            },

            {
                "name": "Layer Normalization",
                "normalize_over": "Feature dimension",
                "formula": "(x - μ_layer) / σ_layer",
                "used_in": "Transformers, RNNs, NLP",
                "pros": "Independent of batch size, works well for sequences",
                "cons": "Slightly more computation than BN"
            },

            {
                "name": "RMS Normalization (Llama)",
                "normalize_over": "Feature dimension",
                "formula": "x / RMS(x), where RMS = sqrt(mean(x²))",
                "used_in": "Llama 2, some modern LLMs",
                "pros": "Simpler (no mean subtraction), faster, similar performance",
                "cons": "Less intuitive"
            }
        ]

        for norm in norms:
            print(f"\n{norm['name']}")
            print(f"  Normalize over: {norm['normalize_over']}")
            print(f"  Formula: {norm['formula']}")
            print(f"  Used in: {norm['used_in']}")
            print(f"  Pros: {norm['pros']}")
            print(f"  Cons: {norm['cons']}")

    @staticmethod
    def demonstrate_residual_benefits():
        """
        Show why residual connections are critical
        """
        print("\n" + "="*80)
        print("WHY RESIDUAL CONNECTIONS?")
        print("="*80)
        print("""
Problem: Training Deep Networks

Without Residuals:
  Layer 1 → Layer 2 → ... → Layer 100
  - Gradients vanish/explode
  - Hard to train beyond 10-20 layers
  - Information loss through layers

With Residuals (x + Layer(x)):
  ✅ Gradient Highway
     - Gradients flow directly through skip connections
     - ∇L/∂x = ∇L/∂output × (1 + ∇Layer/∂x)
     - Always at least gradient of 1!

  ✅ Identity Mapping
     - If Layer learns 0, output = x (no harm)
     - Network can learn when NOT to transform

  ✅ Ensemble Effect
     - Each layer contributes residual adjustment
     - Like ensemble of shallow networks

Result: Can train 100+ layer networks
  - GPT-3: 96 layers
  - PaLM: 118 layers
  - All possible thanks to residuals!
        """)


# Demo
if __name__ == "__main__":
    print("="*80)
    print("LAYER NORMALIZATION & RESIDUAL CONNECTIONS")
    print("="*80)

    d_model = 512
    seq_len = 20
    batch_size = 2

    # Layer Normalization
    print("\n--- Layer Normalization ---")
    ln = LayerNorm(d_model)

    x = torch.randn(batch_size, seq_len, d_model) * 10  # Large variance
    output = ln(x)

    print(f"Input statistics:")
    print(f"  Mean: {x.mean():.4f}")
    print(f"  Std: {x.std():.4f}")
    print(f"  Min: {x.min():.4f}")
    print(f"  Max: {x.max():.4f}")

    print(f"\nOutput statistics (after LN):")
    print(f"  Mean: {output.mean():.4f}")
    print(f"  Std: {output.std():.4f}")
    print(f"  Min: {output.min():.4f}")
    print(f"  Max: {output.max():.4f}")

    # Residual Connection
    print("\n--- Residual Connection ---")
    residual = ResidualConnection(d_model, dropout=0.1)

    # Dummy sublayer
    class DummyLayer(nn.Module):
        def forward(self, x):
            return x * 0.1  # Small transformation

    sublayer = DummyLayer()

    x = torch.randn(batch_size, seq_len, d_model)
    output = residual(x, sublayer)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output ≈ Input? {torch.allclose(output, x, atol=0.5)}")
    print(f"  (Should be similar due to residual connection)")

    # Comparisons
    NormalizationComparison.explain_normalizations()
    NormalizationComparison.demonstrate_residual_benefits()
```

*[Suite avec architecture Transformer complète et comparaisons dans la partie 3...]*

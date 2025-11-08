# Chapitre 2: Architectures Transformers et Self-Attention

## Introduction

Le **Transformer** (2017) a révolutionné le NLP et l'IA. Avant les Transformers, les RNNs et LSTMs dominaient mais souffraient de limitations majeures.

### Pourquoi les Transformers ont tout changé?

```python
"""
Problèmes des RNNs/LSTMs (avant 2017):

1. Sequential Processing
   - Process tokens un par un: t1 → t2 → t3 → ... → tn
   - IMPOSSIBLE à paralléliser
   - Training très lent (jours/semaines)

2. Vanishing Gradients
   - Information à long terme se perd
   - Difficulté avec longues séquences (>100 tokens)

3. No Direct Long-Range Dependencies
   - Pour connecter token 1 et token 100:
     * Information doit passer par 99 étapes
     * Chaque étape = risque de perte d'info

Solution: Transformers avec Self-Attention

1. Parallel Processing
   - Tous les tokens traités simultanément
   - Training 100-1000x plus rapide

2. Direct Connections
   - Chaque token peut "voir" TOUS les autres tokens
   - En une seule opération (attention)

3. Scalability
   - Peut gérer 100k+ tokens (context window)
   - Scales avec GPU/TPU

Résultat: GPT, BERT, T5, Claude, Llama... TOUS basés sur Transformers
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

class ArchitectureType(Enum):
    """Types d'architectures Transformer"""
    ENCODER_ONLY = "encoder_only"      # BERT, RoBERTa
    DECODER_ONLY = "decoder_only"      # GPT, Llama, Claude
    ENCODER_DECODER = "encoder_decoder"  # T5, BART

@dataclass
class ArchitectureComparison:
    """Comparaison des architectures"""
    name: str
    type: ArchitectureType
    use_case: str
    attention_type: str
    examples: list[str]
    strengths: str

# Comparaison des 3 types
ARCHITECTURES = [
    ArchitectureComparison(
        name="Encoder-Only (BERT)",
        type=ArchitectureType.ENCODER_ONLY,
        use_case="Classification, NER, Q&A",
        attention_type="Bidirectional (voit passé + futur)",
        examples=["BERT", "RoBERTa", "ALBERT", "DeBERTa"],
        strengths="Comprend contexte complet, excellent pour classification"
    ),

    ArchitectureComparison(
        name="Decoder-Only (GPT)",
        type=ArchitectureType.DECODER_ONLY,
        use_case="Text Generation, Chat, Code",
        attention_type="Causal (voit seulement passé)",
        examples=["GPT-3", "GPT-4", "Llama 2", "Claude", "Mistral"],
        strengths="Génération de texte fluide, autoregressive"
    ),

    ArchitectureComparison(
        name="Encoder-Decoder (T5)",
        type=ArchitectureType.ENCODER_DECODER,
        use_case="Translation, Summarization",
        attention_type="Encoder bidirectional + Decoder causal",
        examples=["T5", "BART", "mT5", "Flan-T5"],
        strengths="Excellent pour seq2seq tasks"
    )
]

print("="*80)
print("TRANSFORMER ARCHITECTURES COMPARISON")
print("="*80)
for arch in ARCHITECTURES:
    print(f"\n{arch.name}")
    print(f"  Use Case: {arch.use_case}")
    print(f"  Attention: {arch.attention_type}")
    print(f"  Examples: {', '.join(arch.examples)}")
    print(f"  Strengths: {arch.strengths}")
```

## 1. Self-Attention: Le Mécanisme Clé

```python
"""
Self-Attention = Mécanisme qui permet à chaque token d'attendre (pay attention)
à tous les autres tokens de la séquence

Exemple:
Phrase: "The cat sat on the mat"
Token: "it"

Q: À quoi "it" fait référence?
Self-Attention: Calcule scores pour chaque mot
  - "The" → 0.05
  - "cat" → 0.85  ← HIGH SCORE!
  - "sat" → 0.02
  - "on" → 0.01
  - "the" → 0.03
  - "mat" → 0.04

Résultat: "it" ≈ "cat" (0.85 weight)

Comment ça marche?

3 étapes:
1. Query (Q): "Qu'est-ce que je cherche?"
2. Key (K): "Qu'est-ce que je représente?"
3. Value (V): "Quelle information je porte?"

Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (base du Transformer)

    Formula:
        Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    Args:
        Q: Query matrix (batch, seq_len, d_k)
        K: Key matrix (batch, seq_len, d_k)
        V: Value matrix (batch, seq_len, d_v)

    Returns:
        Output: Weighted sum of values (batch, seq_len, d_v)
        Attention weights: (batch, seq_len, seq_len)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (batch, seq_len, d_k)
        key: torch.Tensor,    # (batch, seq_len, d_k)
        value: torch.Tensor,  # (batch, seq_len, d_v)
        mask: Optional[torch.Tensor] = None  # (batch, seq_len, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention

        Steps:
            1. Compute attention scores: Q·K^T
            2. Scale by √d_k (prevent vanishing gradients)
            3. Apply mask (for causal attention)
            4. Softmax to get attention weights
            5. Apply dropout
            6. Weighted sum of values

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask (optional)

        Returns:
            output: Attention output
            attention_weights: Attention distribution
        """
        # d_k = dimension of keys/queries
        d_k = query.size(-1)

        # Step 1: Compute attention scores
        # Q·K^T → (batch, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Step 2: Scale by √d_k
        # Why? Prevent dot products from growing too large
        # Large dot products → extreme softmax → vanishing gradients
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask (if provided)
        # For causal attention (GPT): mask future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 4: Softmax to get attention weights
        # Converts scores to probability distribution
        attention_weights = F.softmax(scores, dim=-1)

        # Step 5: Apply dropout (regularization)
        attention_weights = self.dropout(attention_weights)

        # Step 6: Weighted sum of values
        # (batch, seq_len, seq_len) × (batch, seq_len, d_v)
        # → (batch, seq_len, d_v)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class SelfAttentionExplainer:
    """
    Explain self-attention with concrete example
    """

    @staticmethod
    def explain_with_example():
        """
        Walkthrough self-attention computation
        """
        print("\n" + "="*80)
        print("SELF-ATTENTION WALKTHROUGH")
        print("="*80)

        # Example sentence
        sentence = "The cat sat on the mat"
        tokens = sentence.split()
        seq_len = len(tokens)
        d_model = 4  # Small dimension for clarity

        print(f"\nSentence: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"Sequence length: {seq_len}")
        print(f"Embedding dimension: {d_model}")

        # Random embeddings (in practice, from embedding layer)
        np.random.seed(42)
        embeddings = np.random.randn(seq_len, d_model)

        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embeddings:\n{embeddings.round(2)}")

        # Linear projections to Q, K, V
        # In practice, these are learned weight matrices
        W_q = np.random.randn(d_model, d_model)
        W_k = np.random.randn(d_model, d_model)
        W_v = np.random.randn(d_model, d_model)

        Q = embeddings @ W_q
        K = embeddings @ W_k
        V = embeddings @ W_v

        print(f"\nQuery (Q) shape: {Q.shape}")
        print(f"Key (K) shape: {K.shape}")
        print(f"Value (V) shape: {V.shape}")

        # Compute attention scores
        scores = Q @ K.T  # (seq_len, seq_len)
        print(f"\nAttention scores (before scaling):\n{scores.round(2)}")

        # Scale
        d_k = d_model
        scores_scaled = scores / np.sqrt(d_k)
        print(f"\nAttention scores (after scaling by √{d_k}):\n{scores_scaled.round(2)}")

        # Softmax
        exp_scores = np.exp(scores_scaled - scores_scaled.max(axis=-1, keepdims=True))
        attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        print(f"\nAttention weights (after softmax):")
        print(f"Shape: {attention_weights.shape}")
        print(f"(Each row sums to 1.0)")
        print("\nWeights matrix:")
        print("        ", "  ".join(f"{t:>5s}" for t in tokens))
        for i, token in enumerate(tokens):
            weights_str = "  ".join(f"{w:5.2f}" for w in attention_weights[i])
            print(f"{token:>6s}: [{weights_str}]")

        # Weighted sum of values
        output = attention_weights @ V
        print(f"\nOutput shape: {output.shape}")
        print(f"Output:\n{output.round(2)}")

        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nFor each token, attention weights show:")
        print("  - How much it 'attends' to other tokens")
        print("  - Higher weight = more relevant context")
        print("\nExample: Token 'cat' (index 1)")
        cat_idx = 1
        print(f"Attention weights: {attention_weights[cat_idx].round(3)}")
        most_attended = np.argmax(attention_weights[cat_idx])
        print(f"Most attended token: '{tokens[most_attended]}' "
              f"(weight: {attention_weights[cat_idx][most_attended]:.3f})")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("SCALED DOT-PRODUCT ATTENTION")
    print("="*80)

    # Create attention module
    attention = ScaledDotProductAttention(dropout=0.1)

    # Example input
    batch_size = 2
    seq_len = 5
    d_k = 64
    d_v = 64

    # Random Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)

    print(f"\nInput shapes:")
    print(f"  Query (Q): {Q.shape}")
    print(f"  Key (K): {K.shape}")
    print(f"  Value (V): {V.shape}")

    # Forward pass
    output, weights = attention(Q, K, V)

    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {weights.shape}")

    print(f"\nAttention weights statistics:")
    print(f"  Min: {weights.min():.4f}")
    print(f"  Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Sum per row: {weights.sum(dim=-1)[0, 0]:.4f} (should be 1.0)")

    # Detailed example
    explainer = SelfAttentionExplainer()
    explainer.explain_with_example()
```

## 2. Multi-Head Attention

```python
"""
Multi-Head Attention = Plusieurs self-attention en parallèle

Pourquoi?

Single attention head:
  - Peut capturer 1 type de relation
  - Exemple: "cat" → "mat" (location)

Multi-head (8 heads):
  - Head 1: Relations syntaxiques (sujet-verbe)
  - Head 2: Relations sémantiques (synonymes)
  - Head 3: Coréférences (it → cat)
  - Head 4: Long-range dependencies
  - ... etc

Résultat: Modèle plus expressif, capture patterns multiples

Architecture:
  Input → Split into h heads → h × Attention → Concat → Linear → Output

Formula:
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
  where head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
"""

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (from "Attention Is All You Need")

    Runs multiple attention heads in parallel, then concatenates.

    Args:
        d_model: Model dimension (e.g., 512, 768, 1024)
        num_heads: Number of attention heads (e.g., 8, 12, 16)
        dropout: Dropout rate

    Example:
        d_model = 512, num_heads = 8
        → Each head: d_k = d_v = 512 / 8 = 64
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Note: Single matrix for all heads (more efficient)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Attention module
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split last dimension into (num_heads, d_k)

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()

        # Reshape: (batch, seq_len, d_model)
        #       → (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch, seq_len, num_heads, d_k)
        #         → (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads

        Args:
            x: (batch, num_heads, seq_len, d_k)

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # Transpose: (batch, num_heads, seq_len, d_k)
        #         → (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)

        # Reshape: (batch, seq_len, num_heads, d_k)
        #       → (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,  # (batch, seq_len, d_model)
        key: torch.Tensor,    # (batch, seq_len, d_model)
        value: torch.Tensor,  # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask

        Returns:
            output: Attention output
            attention_weights: Attention weights (for visualization)
        """
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Apply attention for each head
        # Mask shape: (batch, 1, seq_len, seq_len) for broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension

        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch, num_heads, seq_len, d_k)
        # attn_weights: (batch, num_heads, seq_len, seq_len)

        # 4. Concatenate heads
        output = self.combine_heads(attn_output)
        # (batch, seq_len, d_model)

        # 5. Final linear projection
        output = self.W_o(output)

        # 6. Dropout
        output = self.dropout(output)

        return output, attn_weights


class MultiHeadAttentionVisualizer:
    """
    Visualize multi-head attention patterns
    """

    @staticmethod
    def demonstrate_heads():
        """
        Show how different heads capture different patterns
        """
        print("\n" + "="*80)
        print("MULTI-HEAD ATTENTION: WHY MULTIPLE HEADS?")
        print("="*80)

        example_patterns = [
            {
                "head": 1,
                "pattern": "Syntactic (Subject-Verb)",
                "example": "The cat [sat] → strong attention to 'cat' (subject)",
                "learns": "Grammatical structure"
            },
            {
                "head": 2,
                "pattern": "Semantic (Similar Words)",
                "example": "'happy' → strong attention to 'joyful', 'excited'",
                "learns": "Word meanings and synonyms"
            },
            {
                "head": 3,
                "pattern": "Positional (Adjacent Words)",
                "example": "Token n → strong attention to n-1, n+1",
                "learns": "Local context"
            },
            {
                "head": 4,
                "pattern": "Long-Range Dependencies",
                "example": "'it' at position 50 → 'cat' at position 5",
                "learns": "Distant relationships"
            },
            {
                "head": 5,
                "pattern": "Coreference",
                "example": "'he', 'she', 'it' → their antecedents",
                "learns": "Pronoun resolution"
            },
            {
                "head": 6,
                "pattern": "Entity Relations",
                "example": "'Paris' → 'France', 'Eiffel Tower'",
                "learns": "World knowledge"
            },
        ]

        for head in example_patterns:
            print(f"\nHead {head['head']}: {head['pattern']}")
            print(f"  Example: {head['example']}")
            print(f"  Learns: {head['learns']}")

        print("\n" + "="*80)
        print("BENEFITS")
        print("="*80)
        print("""
1. Richer Representations
   - Single head: 1 type of relationship
   - 8 heads: 8 different perspectives

2. Specialization
   - Each head can specialize in different patterns
   - Emerges naturally during training

3. Robustness
   - If one head fails, others compensate
   - More stable learning

4. Parallelism
   - All heads computed simultaneously
   - No sequential dependency
        """)


# Demo
if __name__ == "__main__":
    print("="*80)
    print("MULTI-HEAD ATTENTION")
    print("="*80)

    # Model configuration (like GPT-2 small)
    d_model = 768
    num_heads = 12
    seq_len = 10
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k per head: {d_model // num_heads}")

    # Create module
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.1)

    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Example input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, attn_weights = mha(x, x, x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"  → {num_heads} heads × (seq_len={seq_len} × seq_len={seq_len})")

    # Analyze attention patterns
    print(f"\nAttention statistics (head 0):")
    head_0_weights = attn_weights[0, 0]  # First batch, first head
    print(f"  Shape: {head_0_weights.shape}")
    print(f"  Min: {head_0_weights.min():.4f}")
    print(f"  Max: {head_0_weights.max():.4f}")
    print(f"  Mean: {head_0_weights.mean():.4f}")

    # Visualize
    visualizer = MultiHeadAttentionVisualizer()
    visualizer.demonstrate_heads()
```

*[Suite avec Positional Encoding et architecture complète dans la partie 2...]*

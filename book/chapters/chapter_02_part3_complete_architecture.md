# Chapitre 2 (Partie 3): Architecture Transformer Complète

## 6. Transformer Block (Encoder Layer)

```python
"""
Transformer Encoder Block = Assemblage des composants

Architecture:
  Input
    ↓
  ┌─────────────────────┐
  │ Multi-Head Attention│ ← Self-attention
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │ Add & Norm          │ ← Residual + LayerNorm
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │ Feed-Forward        │ ← FFN
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │ Add & Norm          │ ← Residual + LayerNorm
  └──────────┬──────────┘
             ↓
  Output

Deux sous-couches:
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward

Chacune avec residual connection + layer norm
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy

# Import des modules précédents
# (Dans un vrai projet, ces imports viendraient des fichiers précédents)
from dataclasses import dataclass


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer

    Consists of:
      1. Multi-Head Self-Attention
      2. Position-wise Feed-Forward Network
    Each with residual connection and layer normalization.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        activation: Activation function for FFN
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-Forward Network
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )

        # Layer Normalization (Pre-LN)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder layer

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # 1. Multi-Head Self-Attention with Pre-LN
        # Normalize first
        x_norm = self.norm1(x)

        # Self-attention (Q = K = V = x)
        attn_output, attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            mask=mask
        )

        # Residual connection
        x = x + self.dropout(attn_output)

        # 2. Feed-Forward Network with Pre-LN
        # Normalize
        x_norm = self.norm2(x)

        # FFN
        ffn_output = self.ffn(x_norm)

        # Residual connection
        x = x + self.dropout(ffn_output)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of N Transformer Encoder Layers

    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation for FFN
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Create N identical layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all encoder layers

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            output: Final encoder output
            attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []

        # Pass through each layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attention_weights.append(attn_weights)

        # Final normalization
        x = self.norm(x)

        return x, all_attention_weights


## 7. Decoder Block et Architecture Complète

class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer

    Consists of:
      1. Masked Multi-Head Self-Attention (causal)
      2. Cross-Attention to encoder output
      3. Position-wise Feed-Forward Network

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation for FFN
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 2. Cross-Attention (to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 3. Feed-Forward Network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)

        # Layer Norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder layer

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Encoder output (batch, src_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target (causal) mask

        Returns:
            output: Decoder output
            self_attn_weights: Self-attention weights
            cross_attn_weights: Cross-attention weights
        """
        # 1. Masked Self-Attention
        x_norm = self.norm1(x)
        self_attn_output, self_attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            mask=tgt_mask  # Causal mask
        )
        x = x + self.dropout(self_attn_output)

        # 2. Cross-Attention to encoder
        x_norm = self.norm2(x)
        cross_attn_output, cross_attn_weights = self.cross_attn(
            query=x_norm,  # From decoder
            key=encoder_output,  # From encoder
            value=encoder_output,  # From encoder
            mask=src_mask
        )
        x = x + self.dropout(cross_attn_output)

        # 3. Feed-Forward
        x_norm = self.norm3(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)

        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Stack of N Transformer Decoder Layers

    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation for FFN
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Forward pass through all decoder layers

        Returns:
            output: Final decoder output
            self_attention_weights: List of self-attention weights
            cross_attention_weights: List of cross-attention weights
        """
        all_self_attn = []
        all_cross_attn = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, encoder_output, src_mask, tgt_mask
            )
            all_self_attn.append(self_attn)
            all_cross_attn.append(cross_attn)

        x = self.norm(x)

        return x, all_self_attn, all_cross_attn


# Demo
if __name__ == "__main__":
    print("="*80)
    print("TRANSFORMER ENCODER-DECODER ARCHITECTURE")
    print("="*80)

    # Configuration (similar to Transformer base model)
    config = {
        "num_layers": 6,
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create encoder
    encoder = TransformerEncoder(**config)

    # Create decoder
    decoder = TransformerDecoder(**config)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())

    print(f"\nParameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total: {encoder_params + decoder_params:,}")

    # Example input
    batch_size = 2
    src_len = 20
    tgt_len = 15

    src = torch.randn(batch_size, src_len, config["d_model"])
    tgt = torch.randn(batch_size, tgt_len, config["d_model"])

    print(f"\nExample forward pass:")
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")

    # Encoder
    enc_output, enc_attn_weights = encoder(src)
    print(f"  Encoder output: {enc_output.shape}")
    print(f"  Encoder attention layers: {len(enc_attn_weights)}")

    # Decoder
    dec_output, self_attn, cross_attn = decoder(tgt, enc_output)
    print(f"  Decoder output: {dec_output.shape}")
    print(f"  Decoder self-attention layers: {len(self_attn)}")
    print(f"  Decoder cross-attention layers: {len(cross_attn)}")
```

## 8. Causal Mask (pour GPT-style models)

```python
"""
Causal Mask = Empêche de voir le futur

Pour autoregressive generation (GPT):
  - Token à position i ne peut voir que positions ≤ i
  - Empêche "cheating" pendant training

Exemple:
  Phrase: "The cat sat"

  Position 0 ("The"): voit seulement "The"
  Position 1 ("cat"): voit "The", "cat"
  Position 2 ("sat"): voit "The", "cat", "sat"

Mask matrix (1 = allowed, 0 = blocked):
       The  cat  sat
  The  [1    0    0]
  cat  [1    1    0]
  sat  [1    1    1]

→ Lower triangular matrix
"""

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal (autoregressive) mask

    Prevents attention to future positions.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape (seq_len, seq_len)
        - 1: Allowed to attend
        - 0: Blocked (will be set to -inf in attention)
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))

    return mask


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask

    Prevents attention to padding tokens.

    Args:
        seq: Token IDs (batch, seq_len)
        pad_idx: Padding token ID

    Returns:
        Mask of shape (batch, 1, seq_len)
        - 1: Real token
        - 0: Padding token
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (seq != pad_idx).unsqueeze(1)

    return mask


def create_combined_mask(
    tgt: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Create combined causal + padding mask for decoder

    Args:
        tgt: Target token IDs (batch, tgt_len)
        pad_idx: Padding token ID

    Returns:
        Combined mask (batch, 1, tgt_len, tgt_len)
    """
    batch_size, tgt_len = tgt.size()

    # Causal mask
    causal_mask = create_causal_mask(tgt_len)  # (tgt_len, tgt_len)

    # Padding mask
    padding_mask = create_padding_mask(tgt, pad_idx)  # (batch, 1, tgt_len)

    # Combine: broadcast and element-wise AND
    # (1, tgt_len, tgt_len) & (batch, 1, tgt_len) → (batch, tgt_len, tgt_len)
    combined_mask = causal_mask.unsqueeze(0) & padding_mask.unsqueeze(2)

    # Add dimension for multi-head: (batch, 1, tgt_len, tgt_len)
    combined_mask = combined_mask.unsqueeze(1)

    return combined_mask


class MaskVisualizer:
    """
    Visualize attention masks
    """

    @staticmethod
    def visualize_causal_mask(seq_len: int = 8):
        """
        Print causal mask for understanding
        """
        print("\n" + "="*80)
        print("CAUSAL MASK VISUALIZATION")
        print("="*80)

        mask = create_causal_mask(seq_len)

        print(f"\nSequence length: {seq_len}")
        print(f"Mask shape: {mask.shape}")
        print("\nMask matrix (1 = can attend, 0 = blocked):")
        print("     ", end="")
        for i in range(seq_len):
            print(f"t{i}  ", end="")
        print()

        for i in range(seq_len):
            print(f"t{i}:  ", end="")
            for j in range(seq_len):
                print(f"{int(mask[i, j])}   ", end="")
            print()

        print("\nInterpretation:")
        print(f"  - Row i = What token i can see")
        print(f"  - t0 sees only t0")
        print(f"  - t1 sees t0, t1")
        print(f"  - t{seq_len-1} sees all tokens")
        print(f"  → Lower triangular matrix")

    @staticmethod
    def explain_mask_usage():
        """
        Explain when to use which mask
        """
        print("\n" + "="*80)
        print("MASK TYPES AND USAGE")
        print("="*80)

        masks = [
            {
                "type": "Causal Mask",
                "shape": "(seq_len, seq_len)",
                "purpose": "Prevent seeing future tokens",
                "used_in": "GPT (decoder-only), autoregressive models",
                "example": "Lower triangular matrix"
            },

            {
                "type": "Padding Mask",
                "shape": "(batch, seq_len)",
                "purpose": "Ignore padding tokens",
                "used_in": "All models with variable-length sequences",
                "example": "[1,1,1,1,0,0] for seq with 2 padding"
            },

            {
                "type": "Attention Mask (Encoder)",
                "shape": "(batch, seq_len, seq_len)",
                "purpose": "Bidirectional attention + padding",
                "used_in": "BERT, encoder-only models",
                "example": "All 1s except padding rows/cols"
            },

            {
                "type": "Combined Mask (Decoder)",
                "shape": "(batch, tgt_len, tgt_len)",
                "purpose": "Causal + padding",
                "used_in": "GPT during training, T5 decoder",
                "example": "Lower triangular + padding"
            }
        ]

        for mask in masks:
            print(f"\n{mask['type']}")
            print(f"  Shape: {mask['shape']}")
            print(f"  Purpose: {mask['purpose']}")
            print(f"  Used in: {mask['used_in']}")
            print(f"  Example: {mask['example']}")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("ATTENTION MASKS")
    print("="*80)

    # Causal mask
    print("\n--- Causal Mask ---")
    causal_mask = create_causal_mask(seq_len=5)
    print(f"Shape: {causal_mask.shape}")
    print(f"Mask:\n{causal_mask.int()}")

    # Padding mask
    print("\n--- Padding Mask ---")
    # Example: batch of 2 sequences with padding (0 = padding)
    seq = torch.tensor([
        [1, 2, 3, 4, 0, 0],  # 4 tokens + 2 padding
        [1, 2, 0, 0, 0, 0]   # 2 tokens + 4 padding
    ])
    padding_mask = create_padding_mask(seq, pad_idx=0)
    print(f"Sequences:\n{seq}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Mask:\n{padding_mask.squeeze(1).int()}")

    # Combined mask
    print("\n--- Combined Mask (Causal + Padding) ---")
    combined_mask = create_combined_mask(seq, pad_idx=0)
    print(f"Shape: {combined_mask.shape}")
    print(f"First sequence mask:\n{combined_mask[0, 0].int()}")
    print(f"Second sequence mask:\n{combined_mask[1, 0].int()}")

    # Visualizations
    visualizer = MaskVisualizer()
    visualizer.visualize_causal_mask(seq_len=8)
    visualizer.explain_mask_usage()
```

## 9. Architecture Comparison: GPT vs BERT vs T5

```python
"""
Comparaison des 3 architectures principales
"""

@dataclass
class ModelArchitectureSpec:
    """Specification d'une architecture de modèle"""
    name: str
    type: str
    attention: str
    use_case: str
    encoder_layers: int
    decoder_layers: int
    total_layers: int
    bidirectional: bool
    examples: list[str]
    key_innovation: str


ARCHITECTURES_COMPARISON = [
    ModelArchitectureSpec(
        name="BERT",
        type="Encoder-Only",
        attention="Bidirectional (Full attention)",
        use_case="Classification, NER, Q&A, Understanding",
        encoder_layers=12,  # BERT-base
        decoder_layers=0,
        total_layers=12,
        bidirectional=True,
        examples=["BERT-base (110M)", "BERT-large (340M)", "RoBERTa", "DeBERTa"],
        key_innovation="Masked Language Modeling (MLM)"
    ),

    ModelArchitectureSpec(
        name="GPT",
        type="Decoder-Only",
        attention="Causal (Only past tokens)",
        use_case="Text Generation, Chat, Code, Completion",
        encoder_layers=0,
        decoder_layers=12,  # GPT-2 small
        total_layers=12,
        bidirectional=False,
        examples=["GPT-2 (117M-1.5B)", "GPT-3 (175B)", "GPT-4", "Llama 2", "Claude"],
        key_innovation="Autoregressive generation"
    ),

    ModelArchitectureSpec(
        name="T5",
        type="Encoder-Decoder",
        attention="Encoder: Bidirectional, Decoder: Causal",
        use_case="Translation, Summarization, Seq2Seq",
        encoder_layers=12,  # T5-base
        decoder_layers=12,
        total_layers=24,
        bidirectional=True,  # In encoder
        examples=["T5-base (220M)", "T5-large (770M)", "BART", "mT5"],
        key_innovation="Text-to-Text framework"
    )
]


class ArchitectureComparator:
    """
    Compare different Transformer architectures
    """

    @staticmethod
    def print_comparison():
        """
        Print detailed comparison
        """
        print("="*80)
        print("TRANSFORMER ARCHITECTURES COMPARISON")
        print("="*80)

        for arch in ARCHITECTURES_COMPARISON:
            print(f"\n{'='*80}")
            print(f"{arch.name} ({arch.type})")
            print('='*80)
            print(f"Architecture: {arch.type}")
            print(f"Attention: {arch.attention}")
            print(f"Layers: {arch.encoder_layers} encoder + {arch.decoder_layers} decoder = {arch.total_layers} total")
            print(f"Bidirectional: {arch.bidirectional}")
            print(f"Use Case: {arch.use_case}")
            print(f"Key Innovation: {arch.key_innovation}")
            print(f"Examples: {', '.join(arch.examples)}")

        print("\n" + "="*80)
        print("WHICH TO USE?")
        print("="*80)
        print("""
✅ BERT (Encoder-Only)
  When: Classification, entity extraction, understanding tasks
  Why: Bidirectional context, sees full sentence
  Examples: Sentiment analysis, NER, Q&A (extractive)

✅ GPT (Decoder-Only)
  When: Text generation, chat, code, creative writing
  Why: Autoregressive, natural for generation
  Examples: ChatGPT, code completion, story writing

✅ T5 (Encoder-Decoder)
  When: Seq2Seq tasks, translation, summarization
  Why: Separate encoding and decoding, flexible
  Examples: Translation, summarization, paraphrasing

🚀 Modern Trend: Decoder-Only (GPT-style) dominates!
  - GPT-3, GPT-4, Llama 2, Claude, Mistral all decoder-only
  - Reason: Can do BOTH understanding AND generation
  - Scaling laws favor decoder-only
        """)


# Demo
if __name__ == "__main__":
    comparator = ArchitectureComparator()
    comparator.print_comparison()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Self-Attention = Core innovation
   - Parallel processing
   - Direct long-range connections
   - O(n²) complexity but worth it

2. Multi-Head Attention = Multiple perspectives
   - Different heads learn different patterns
   - Richer representations

3. Positional Encoding = Position awareness
   - Sinusoidal (extrapolate) or Learned (flexible)

4. Feed-Forward = Position-wise processing
   - 4x expansion typical
   - Most parameters here

5. Residual Connections = Enable deep networks
   - Gradient flow
   - 100+ layers possible

6. Layer Normalization = Training stability
   - Pre-LN now standard

7. Three architectures:
   - Encoder-only (BERT): Understanding
   - Decoder-only (GPT): Generation
   - Encoder-decoder (T5): Seq2Seq

8. Modern trend: Decoder-only dominates
   - GPT, Llama, Claude all decoder-only
   - Can do both understanding and generation
    """)
```

*[Suite avec implémentation complète d'un Transformer from scratch dans la partie 4...]*

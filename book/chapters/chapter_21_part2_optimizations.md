# Chapitre 21 - Partie 2: Optimisations pour Long Context

## Techniques d'Optimisation de l'Attention

Pour gérer efficacement de très longs contextes (100k+ tokens), plusieurs techniques permettent de réduire la complexité ou la mémoire nécessaire.

```python
"""
OPTIMISATIONS POUR LONG CONTEXT

Problème fondamental:
  Standard attention = O(n²) en temps et mémoire
  100k tokens → 10 milliards d'opérations + 95GB RAM

Solutions:

1. FLASH ATTENTION (Dao et al. 2022):
   • Réorganise calculs pour réduire accès mémoire
   • Pas d'approximation (exact)
   • 2-4x plus rapide, 10-20x moins de mémoire
   • Algorithme: Tiling + recomputation

2. SPARSE ATTENTION:
   • Chaque token n'attends que sur subset des autres
   • Patterns: local, strided, random
   • O(n √n) ou O(n log n) selon pattern

3. SLIDING WINDOW ATTENTION:
   • Chaque token attends seulement sur W voisins
   • O(n × W) au lieu de O(n²)
   • Utilisé: Mistral (window=4096), Longformer

4. MULTI-QUERY ATTENTION (MQA):
   • Partage key/value entre heads
   • Réduit KV cache de 8x (pour 8 heads)
   • Utilisé: PaLM, Falcon

5. GROUPED-QUERY ATTENTION (GQA):
   • Compromis entre MHA et MQA
   • Groupes de heads partagent KV
   • Utilisé: LLaMA 2, Mistral

6. COMPRESSION DE CONTEXTE:
   • Résumer anciennes parties du contexte
   • Techniques: LongMem, AutoCompressors
   • Trade-off: vitesse vs précision
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math


# ============================================================================
# FLASH ATTENTION
# ============================================================================

"""
FLASH ATTENTION = Attention exacte mais optimisée

Problème standard attention:
  1. Calculer Q @ K^T → (N, N) matrix en HBM (slow memory)
  2. Appliquer softmax
  3. Multiplier par V

  Problème: Matrice (N, N) trop grande pour SRAM (fast memory)
  → Beaucoup d'accès lents à HBM

Solution Flash Attention:
  1. Diviser Q, K, V en blocs (tiles)
  2. Calculer attention par blocs en SRAM
  3. Fusion des calculs (pas de matérialisation intermédiaire)
  4. Recomputation au backward (trade compute vs memory)

Algorithme (simplifié):
  - Diviser Q en blocs de taille Bc
  - Diviser K, V en blocs de taille Br
  - Pour chaque bloc de Q:
    - Pour chaque bloc de K, V:
      - Calculer attention partielle
      - Fusionner avec résultats précédents
    - Obtenir output final pour ce bloc de Q

Résultats:
  • Vitesse: 2-4x plus rapide
  • Mémoire: 10-20x moins (sub-quadratic)
  • Exactitude: Identique à standard attention

Implémentation:
  # PyTorch 2.0+
  import torch.nn.functional as F
  output = F.scaled_dot_product_attention(
      query, key, value,
      is_causal=True
  )  # Utilise Flash Attention automatiquement si disponible
"""

@dataclass
class FlashAttentionConfig:
    """Configuration pour Flash Attention"""
    block_size_q: int = 64      # Taille bloc pour queries
    block_size_kv: int = 64     # Taille bloc pour keys/values
    use_recomputation: bool = True  # Recompute au backward


class FlashAttentionSimple:
    """
    Implémentation simplifiée de Flash Attention (conceptuelle)

    En production, utiliser:
      - torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
      - flash_attn package (triton)
      - xformers.ops.memory_efficient_attention
    """

    def __init__(self, config: FlashAttentionConfig):
        self.config = config

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal_mask: bool = True
    ) -> np.ndarray:
        """
        Flash Attention forward (version simplifiée pour démo)

        Args:
            query: (batch, num_heads, seq_len, head_dim)
            key: (batch, num_heads, seq_len, head_dim)
            value: (batch, num_heads, seq_len, head_dim)
            causal_mask: Appliquer masque causal

        Returns:
            output: (batch, num_heads, seq_len, head_dim)
        """
        batch, num_heads, seq_len, head_dim = query.shape

        print(f"\n⚡ Flash Attention")
        print(f"  Sequence length: {seq_len}")
        print(f"  Block size Q: {self.config.block_size_q}")
        print(f"  Block size KV: {self.config.block_size_kv}")

        # Nombre de blocs
        num_blocks_q = math.ceil(seq_len / self.config.block_size_q)
        num_blocks_kv = math.ceil(seq_len / self.config.block_size_kv)

        print(f"  Blocs Q: {num_blocks_q}")
        print(f"  Blocs KV: {num_blocks_kv}")

        # Économie mémoire
        standard_memory = seq_len * seq_len * head_dim * 4  # float32
        flash_memory = (
            self.config.block_size_q * self.config.block_size_kv * head_dim * 4
        )

        print(f"\n💾 Mémoire:")
        print(f"  Standard: {standard_memory / 1e9:.2f} GB")
        print(f"  Flash: {flash_memory / 1e6:.2f} MB (peak)")
        print(f"  Réduction: {standard_memory / flash_memory:.0f}x")

        # En production: implémentation réelle avec tiling
        # Ici on simule juste avec attention standard
        scale = 1.0 / math.sqrt(head_dim)

        # Standard attention (simplifié)
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale

        if causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask

        attention_weights = self._softmax(scores, axis=-1)
        output = np.matmul(attention_weights, value)

        return output

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax stable numériquement"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# SLIDING WINDOW ATTENTION
# ============================================================================

"""
SLIDING WINDOW ATTENTION = Attention locale

Idée: Chaque token attends seulement sur W tokens autour de lui

Exemple avec window=3:
  Token 5 attends sur tokens [2, 3, 4, 5, 6, 7, 8]

  Pattern:
    0 1 2 3 4 5 6 7 8
  0 ■ ■ ■ □ □ □ □ □ □
  1 ■ ■ ■ ■ □ □ □ □ □
  2 ■ ■ ■ ■ ■ □ □ □ □
  3 □ ■ ■ ■ ■ ■ □ □ □
  4 □ □ ■ ■ ■ ■ ■ □ □
  5 □ □ □ ■ ■ ■ ■ ■ □

  ■ = attends, □ = ignore

Complexité:
  • Temps: O(n × W) au lieu de O(n²)
  • Mémoire: O(n × W)

  Pour W=4096 et n=100k:
    Standard: 10B opérations
    Sliding: 410M opérations (24x moins!)

Avantages:
  ✅ Très efficace pour long context
  ✅ Capture bien dépendances locales
  ✅ Peut être combiné avec global attention

Inconvénients:
  ❌ Perd dépendances long-range
  ❌ Nécessite multiples layers pour info globale

Utilisé par:
  • Mistral: window=4096
  • Longformer: window=512 + global attention
  • BigBird: window + random + global
"""

class SlidingWindowAttention:
    """
    Attention avec fenêtre glissante

    Chaque position i attends sur [i-W//2, i+W//2]
    """

    def __init__(self, window_size: int = 512):
        """
        Args:
            window_size: Taille de la fenêtre d'attention
        """
        self.window_size = window_size

    def create_sliding_window_mask(
        self,
        seq_len: int,
        causal: bool = True
    ) -> np.ndarray:
        """
        Créer masque de fenêtre glissante

        Args:
            seq_len: Longueur de séquence
            causal: Si True, attention causale (pas le futur)

        Returns:
            Mask de shape (seq_len, seq_len)
            True = attendre, False = masquer
        """
        # Créer matrice de distances
        positions = np.arange(seq_len)
        distances = np.abs(positions[:, None] - positions[None, :])

        # Masque: distance <= window_size // 2
        mask = distances <= (self.window_size // 2)

        if causal:
            # Masquer le futur
            causal_mask = np.tril(np.ones((seq_len, seq_len)))
            mask = mask & causal_mask.astype(bool)

        return mask

    def apply_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray
    ) -> np.ndarray:
        """
        Attention avec fenêtre glissante

        Args:
            query, key, value: (batch, heads, seq_len, head_dim)

        Returns:
            output: (batch, heads, seq_len, head_dim)
        """
        batch, heads, seq_len, head_dim = query.shape

        # Créer masque
        mask = self.create_sliding_window_mask(seq_len, causal=True)

        print(f"\n🪟 Sliding Window Attention")
        print(f"  Sequence: {seq_len}")
        print(f"  Window: {self.window_size}")

        # Compter éléments attendus
        attended = np.sum(mask)
        total = seq_len * seq_len
        sparsity = 1 - (attended / total)

        print(f"  Attended: {attended:,} / {total:,}")
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Speedup: ~{total/attended:.1f}x")

        # Calculer attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale

        # Appliquer masque
        scores = np.where(mask, scores, -1e9)

        # Softmax et output
        weights = self._softmax(scores, axis=-1)
        output = np.matmul(weights, value)

        return output

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax stable"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def visualize_pattern(self, seq_len: int = 16):
        """Visualise le pattern d'attention"""
        mask = self.create_sliding_window_mask(seq_len, causal=True)

        print(f"\n📊 Pattern d'attention (seq_len={seq_len}, window={self.window_size})")
        print("     ", end="")
        for j in range(seq_len):
            print(f"{j:3d} ", end="")
        print()

        for i in range(seq_len):
            print(f"{i:3d}  ", end="")
            for j in range(seq_len):
                if mask[i, j]:
                    print(" ■  ", end="")
                else:
                    print(" □  ", end="")
            print()


# ============================================================================
# MULTI-QUERY ATTENTION (MQA) ET GROUPED-QUERY ATTENTION (GQA)
# ============================================================================

"""
MULTI-QUERY ATTENTION (MQA)

Problème: KV cache énorme pour long context
  • Standard (Multi-Head Attention):
    - Chaque head a ses propres K, V
    - KV cache: num_heads × seq_len × head_dim
    - Ex: 32 heads × 100k × 128 = 409M paramètres

  • MQA:
    - 1 seule paire K, V partagée entre tous les heads
    - KV cache: 1 × seq_len × (head_dim × num_heads)
    - Réduction: num_heads fois plus petit

  • GQA (Grouped-Query Attention):
    - Compromis: groupes de heads partagent K, V
    - Ex: 32 heads → 4 groupes → 4 paires KV
    - Réduction: num_heads / num_groups

Performances:
  MHA (Multi-Head): Qualité max, KV cache max
  GQA: 90-95% qualité, KV cache réduit 4-8x
  MQA: 85-90% qualité, KV cache min

Utilisé par:
  • MQA: PaLM (540B), Falcon (40B)
  • GQA: LLaMA 2, Mistral, Qwen
"""

@dataclass
class AttentionConfig:
    """Configuration pour différents types d'attention"""
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # None = MHA, 1 = MQA, autre = GQA
    head_dim: int = 128

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads  # MHA par défaut

    @property
    def is_mha(self) -> bool:
        return self.num_kv_heads == self.num_heads

    @property
    def is_mqa(self) -> bool:
        return self.num_kv_heads == 1

    @property
    def is_gqa(self) -> bool:
        return 1 < self.num_kv_heads < self.num_heads


def compare_kv_cache_sizes(seq_len: int = 100000):
    """Compare tailles de KV cache pour différentes architectures"""
    print("\n" + "="*80)
    print("COMPARAISON KV CACHE")
    print("="*80)
    print(f"Sequence length: {seq_len:,}")

    configs = [
        ("MHA (Standard)", AttentionConfig(num_heads=32, num_kv_heads=32)),
        ("GQA (8 groups)", AttentionConfig(num_heads=32, num_kv_heads=4)),
        ("GQA (4 groups)", AttentionConfig(num_heads=32, num_kv_heads=8)),
        ("MQA (1 group)", AttentionConfig(num_heads=32, num_kv_heads=1)),
    ]

    print(f"\n{'Architecture':<20} {'KV Heads':<10} {'Cache Size':<15} {'Reduction':<15}")
    print("-" * 80)

    base_size = None

    for name, config in configs:
        # KV cache size: num_kv_heads × seq_len × head_dim × 2 (K + V) × 2 bytes (float16)
        cache_size = (
            config.num_kv_heads * seq_len * config.head_dim * 2 * 2
        )

        cache_gb = cache_size / (1024 ** 3)

        if base_size is None:
            base_size = cache_size
            reduction = "1x (baseline)"
        else:
            reduction = f"{base_size / cache_size:.1f}x smaller"

        print(f"{name:<20} {config.num_kv_heads:<10} {cache_gb:>14.2f} GB {reduction:<15}")

    print(f"\n💡 Pour 100k tokens:")
    print(f"  • MHA → GQA (4 groups): 4x moins de mémoire")
    print(f"  • MHA → MQA: 32x moins de mémoire")
    print(f"  • Trade-off: Réduction mémoire vs qualité")


# ============================================================================
# COMPRESSION DE CONTEXTE
# ============================================================================

"""
COMPRESSION DE CONTEXTE = Résumer automatiquement contexte ancien

Problème: Même avec optimisations, 1M tokens coûte cher

Solution: Compress les vieux tokens
  1. Garder tokens récents (ex: derniers 8k) en full resolution
  2. Compresser le reste (ex: 100k → 4k tokens)

Techniques:

1. AutoCompressors (Chevalier et al. 2023):
   • Entraîne modèle à créer "summary tokens"
   • Input: Long segment → Output: Quelques tokens compressés
   • Ces tokens remplacent segment original

2. LongMem (Wang et al. 2023):
   • Cache des "memory" tokens pour segments passés
   • Retrieve dynamiquement selon besoin

3. Récursive Summarization:
   • Simple: Demander au LLM de résumer
   • Ex: "Résume les 50k tokens précédents en 2k tokens"
"""

class ContextCompressor:
    """
    Compresse contexte long en résumé court

    Stratégies:
      1. "recent": Garder N tokens récents
      2. "summarize": Résumer en tokens compressés
      3. "hybrid": Récent + résumé de l'ancien
    """

    def __init__(
        self,
        max_context: int = 8192,
        compression_ratio: int = 10
    ):
        """
        Args:
            max_context: Context max à garder
            compression_ratio: Ratio de compression
        """
        self.max_context = max_context
        self.compression_ratio = compression_ratio

    def compress(
        self,
        tokens: List[int],
        strategy: str = "hybrid"
    ) -> List[int]:
        """
        Compresse une séquence de tokens

        Args:
            tokens: Séquence complète
            strategy: "recent", "summarize", ou "hybrid"

        Returns:
            Tokens compressés
        """
        print(f"\n🗜️  Compression de contexte")
        print(f"  Tokens input: {len(tokens):,}")
        print(f"  Strategy: {strategy}")

        if len(tokens) <= self.max_context:
            print(f"  ✅ Pas de compression nécessaire")
            return tokens

        if strategy == "recent":
            # Garder seulement tokens récents
            compressed = tokens[-self.max_context:]

        elif strategy == "summarize":
            # Résumer tout le contexte
            # En production: appeler LLM pour résumer
            target_len = len(tokens) // self.compression_ratio
            compressed = self._simulate_summarization(tokens, target_len)

        elif strategy == "hybrid":
            # Garder récent + résumer ancien
            recent_len = self.max_context // 2

            recent = tokens[-recent_len:]
            old = tokens[:-recent_len]

            # Résumer ancien
            summary_len = len(old) // self.compression_ratio
            summary = self._simulate_summarization(old, summary_len)

            compressed = summary + recent

        else:
            raise ValueError(f"Strategy {strategy} invalide")

        print(f"  Tokens output: {len(compressed):,}")
        print(f"  Compression: {len(tokens) / len(compressed):.1f}x")

        return compressed

    def _simulate_summarization(
        self,
        tokens: List[int],
        target_len: int
    ) -> List[int]:
        """
        Simule résumé (en production: appel LLM)

        Args:
            tokens: Tokens à résumer
            target_len: Longueur cible

        Returns:
            Tokens résumés
        """
        # Simulation: sample uniformément
        step = len(tokens) // target_len
        if step < 1:
            step = 1

        summarized = tokens[::step][:target_len]

        return summarized


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_optimizations():
    """Démo des optimisations"""
    print("="*80)
    print("OPTIMISATIONS POUR LONG CONTEXT")
    print("="*80)

    # 1. Flash Attention
    print("\n1️⃣  FLASH ATTENTION")
    print("-" * 80)

    config = FlashAttentionConfig(block_size_q=64, block_size_kv=64)
    flash = FlashAttentionSimple(config)

    # Simuler tenseurs
    batch, heads, seq_len, head_dim = 1, 8, 4096, 64
    query = np.random.randn(batch, heads, seq_len, head_dim)
    key = np.random.randn(batch, heads, seq_len, head_dim)
    value = np.random.randn(batch, heads, seq_len, head_dim)

    output = flash.forward(query, key, value)

    # 2. Sliding Window
    print("\n\n2️⃣  SLIDING WINDOW ATTENTION")
    print("-" * 80)

    sliding = SlidingWindowAttention(window_size=512)
    sliding.visualize_pattern(seq_len=16)

    # Test performance
    output = sliding.apply_attention(query, key, value)

    # 3. KV Cache comparison
    print("\n\n3️⃣  KV CACHE: MHA vs GQA vs MQA")
    print("-" * 80)

    compare_kv_cache_sizes(seq_len=100000)

    # 4. Context compression
    print("\n\n4️⃣  COMPRESSION DE CONTEXTE")
    print("-" * 80)

    compressor = ContextCompressor(max_context=8192, compression_ratio=10)

    # Simuler long contexte
    long_tokens = list(range(100000))

    for strategy in ["recent", "summarize", "hybrid"]:
        compressed = compressor.compress(long_tokens, strategy=strategy)


if __name__ == "__main__":
    demo_optimizations()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 2")
    print("="*80)
    print("""
1. FLASH ATTENTION
   • Réorganise calculs pour efficacité mémoire
   • 2-4x plus rapide, 10-20x moins de mémoire
   • Exactement identique à standard attention
   • Intégré dans PyTorch 2.0+

2. SLIDING WINDOW
   • O(n × W) au lieu de O(n²)
   • Excellent pour dépendances locales
   • Utilisé: Mistral (4k window), Longformer

3. MULTI-QUERY / GROUPED-QUERY ATTENTION
   MHA: Qualité max, KV cache max
   GQA: 4-8x moins de KV cache, ~95% qualité
   MQA: 32x moins de KV cache, ~85% qualité

4. COMPRESSION DE CONTEXTE
   • Résumer contexte ancien
   • Garder contexte récent
   • Trade-off: tokens vs précision

5. COMBINAISONS GAGNANTES
   Flash Attention + GQA + Sliding Window:
     → 100k tokens possible sur 1x A100
     → Coût raisonnable
     → Qualité préservée

PROCHAINE PARTIE: Évaluation et projets long context
    """)

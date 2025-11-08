# Chapitre 21: Long Context et Gestion de la Mémoire

## Introduction au Long Context

Le **long context** permet aux LLMs de traiter des séquences de 100k, 200k, voire 1M+ tokens, débloquant des applications impossibles avec des contextes courts (2k-8k tokens).

```python
"""
LONG CONTEXT = Capacité à traiter de très longues séquences

Évolution historique:
  2018: GPT-2 (1k tokens)
  2020: GPT-3 (2k → 4k tokens)
  2022: GPT-3.5 (4k tokens)
  2023: GPT-4 (8k → 32k → 128k tokens)
  2024: Gemini 1.5 (1M tokens), Claude 3 (200k tokens)

Pourquoi c'est difficile?
  1. Complexité computationnelle:
     Self-attention = O(n²) où n = longueur séquence
     4k tokens: 16M opérations
     100k tokens: 10 MILLIARDS d'opérations
     → 625x plus lent!

  2. Mémoire:
     Stocker attention matrix: n × n × d
     100k tokens × 100k tokens = 10B éléments
     En float16: ~20GB juste pour l'attention!

  3. Position encoding:
     Comment encoder position 100,000?
     Sinusoidal original: extrapolation mauvaise

Applications du long context:
  ✅ Analyser livres entiers (100k+ tokens)
  ✅ Codebase complet (50k-500k tokens)
  ✅ Conversations très longues (historique complet)
  ✅ Documents légaux/médicaux (100+ pages)
  ✅ Analyse vidéo complète (transcription)

Limitations actuelles:
  ❌ Coût élevé (linéaire avec contexte)
  ❌ "Lost in the middle" (info milieu moins bien utilisée)
  ❌ Latence (plus de tokens = plus lent)
  ❌ Qualité variable selon position
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math


# ============================================================================
# POSITION ENCODINGS: FONDATIONS DU LONG CONTEXT
# ============================================================================

"""
POSITION ENCODING = Comment le modèle sait l'ordre des tokens

Problème: Transformer est permutation-invariant
  ["chat mange souris"] ≈ ["souris mange chat"] sans position info

Solutions:

1. SINUSOIDAL (Vaswani 2017 - original Transformer):
   PE(pos, 2i) = sin(pos / 10000^(2i/d))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

   ✅ Ne nécessite pas d'entraînement
   ✅ Peut extrapoler (théoriquement)
   ❌ Extrapolation réelle mauvaise au-delà de 2x training length
   ❌ Pas appris, donc sous-optimal

2. LEARNED POSITIONAL EMBEDDINGS (GPT-2, BERT):
   Embedding table: positions → vecteurs appris

   ✅ Optimisé pour les données
   ❌ Taille fixe (max_position_embeddings)
   ❌ Pas d'extrapolation possible
   ❌ O(n) paramètres

3. RoPE - Rotary Position Embedding (Su et al. 2021):
   Rotation des query/key dans espace complexe

   ✅ Relative position encoding
   ✅ Extrapolation possible avec techniques
   ✅ Pas de paramètres additionnels
   ✅ Utilisé par: LLaMA, Mistral, Qwen, etc.

4. ALiBi - Attention with Linear Biases (Press et al. 2022):
   Ajoute bias linéaire à l'attention selon distance

   ✅ Extrapolation excellente
   ✅ Très simple
   ✅ Pas de paramètres
   ✅ Utilisé par: BLOOM, MPT

5. PI - Position Interpolation (Chen et al. 2023):
   Interpoler positions au lieu d'extrapoler

   ✅ Étend RoPE de 2k → 32k facilement
   ✅ Peu de fine-tuning nécessaire
"""

class SinusoidalPositionEncoding:
    """
    Position encoding sinusoïdal original (Transformer 2017)

    Formule:
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Propriétés:
      • PE(pos + k) peut être exprimé comme fonction linéaire de PE(pos)
      • Permet au modèle d'apprendre positions relatives
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimension du modèle (doit être pair)
            max_len: Longueur maximum de séquence
        """
        self.d_model = d_model
        self.max_len = max_len

        # Précalculer les position encodings
        self.pe = self._compute_pe()

    def _compute_pe(self) -> np.ndarray:
        """
        Calcule la matrice de position encoding

        Returns:
            Array de shape (max_len, d_model)
        """
        pe = np.zeros((self.max_len, self.d_model))

        position = np.arange(0, self.max_len).reshape(-1, 1)  # (max_len, 1)

        # Calcul des fréquences
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        )  # (d_model/2,)

        # Sin pour indices pairs, cos pour indices impairs
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def encode(self, seq_len: int) -> np.ndarray:
        """
        Retourne position encoding pour une séquence

        Args:
            seq_len: Longueur de la séquence

        Returns:
            Position encodings de shape (seq_len, d_model)
        """
        if seq_len > self.max_len:
            # Extrapolation (mauvaise qualité)
            print(f"⚠️  Extrapolation: seq_len {seq_len} > max_len {self.max_len}")

        return self.pe[:seq_len]

    def visualize_patterns(self):
        """Affiche les patterns du position encoding"""
        print("\n" + "="*80)
        print("SINUSOIDAL POSITION ENCODING")
        print("="*80)
        print(f"d_model: {self.d_model}")
        print(f"max_len: {self.max_len}")

        # Afficher quelques positions
        positions = [0, 10, 100, 1000]
        print(f"\nExemple d'encodings (premières 8 dimensions):")

        for pos in positions:
            if pos < self.max_len:
                encoding = self.pe[pos, :8]
                print(f"  Position {pos:4d}: [{', '.join([f'{x:6.3f}' for x in encoding])}...]")


class RoPE:
    """
    Rotary Position Embedding (RoPE)

    Idée: Rotation dans l'espace complexe pour encoder position relative

    Au lieu d'ajouter position au token embedding, RoPE rotate
    les query et key vectors d'un angle proportionnel à leur position.

    Pour position m, rotation de θ = m × base^(-2i/d) où i = dimension

    Avantages:
      ✅ Encode position relative directement dans l'attention
      ✅ Pas de paramètres additionnels
      ✅ Extrapolation possible avec ajustements
      ✅ Utilisé dans LLaMA, Mistral, Qwen

    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000
    ):
        """
        Args:
            dim: Dimension (head_dim, typiquement 64 ou 128)
            max_seq_len: Longueur max durant training
            base: Base pour les fréquences (10000 par défaut)
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Précalculer les fréquences
        self.freqs = self._compute_freqs()

    def _compute_freqs(self) -> np.ndarray:
        """
        Calcule les fréquences pour chaque dimension

        freq_i = 1 / (base^(2i/dim))
        """
        # Indices de dimension (0, 2, 4, ...)
        indices = np.arange(0, self.dim, 2)

        # Fréquences
        freqs = 1.0 / (self.base ** (indices / self.dim))

        return freqs

    def apply_rotary_emb(
        self,
        x: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Applique rotary embedding à x

        Args:
            x: Tensor de shape (batch, seq_len, dim)
            positions: Positions de shape (seq_len,)

        Returns:
            x avec RoPE appliqué
        """
        # Calculer angles: position * freq pour chaque dimension
        # positions: (seq_len,)
        # freqs: (dim/2,)
        # angles: (seq_len, dim/2)
        angles = np.outer(positions, self.freqs)

        # Créer rotation matrix avec cos et sin
        cos = np.cos(angles)  # (seq_len, dim/2)
        sin = np.sin(angles)

        # Appliquer rotation
        # x est de shape (batch, seq_len, dim)
        # On travaille sur les paires de dimensions

        batch, seq_len, dim = x.shape
        x_rotated = np.zeros_like(x)

        # Rotation pour chaque paire (x[2i], x[2i+1])
        for i in range(dim // 2):
            x1 = x[:, :, 2*i]      # (batch, seq_len)
            x2 = x[:, :, 2*i+1]

            # Rotation 2D
            x_rotated[:, :, 2*i] = x1 * cos[:, i] - x2 * sin[:, i]
            x_rotated[:, :, 2*i+1] = x1 * sin[:, i] + x2 * cos[:, i]

        return x_rotated

    def extend_context(
        self,
        new_max_len: int,
        method: str = "linear_scaling"
    ):
        """
        Étend le contexte au-delà de max_seq_len

        Méthodes:
          1. "linear_scaling": Scale les fréquences linéairement
             (Position Interpolation - Chen et al.)

          2. "ntk_aware": NTK-aware scaling (plus sophistiqué)
             (Reddit: /u/bloc97)

          3. "yarn": YaRN (Yet another RoPE extensioN)
             (Peng et al. 2023)

        Args:
            new_max_len: Nouvelle longueur max
            method: Méthode d'extension
        """
        print(f"\n🔧 Extension RoPE: {self.max_seq_len} → {new_max_len}")
        print(f"   Méthode: {method}")

        scaling_factor = new_max_len / self.max_seq_len

        if method == "linear_scaling":
            # Position Interpolation: diviser fréquences par scaling_factor
            self.freqs = self.freqs / scaling_factor
            print(f"   ✅ Fréquences scaled par 1/{scaling_factor:.2f}")

        elif method == "ntk_aware":
            # NTK-aware: ajuster le base plutôt que les fréquences
            new_base = self.base * (scaling_factor ** (self.dim / (self.dim - 2)))
            self.base = new_base
            self.freqs = self._compute_freqs()
            print(f"   ✅ Base ajusté: {self.base:.0f}")

        self.max_seq_len = new_max_len


class ALiBi:
    """
    Attention with Linear Biases (ALiBi)

    Idée ultra-simple: Ajouter un bias linéaire aux scores d'attention
    basé sur la distance entre query et key.

    bias(i, j) = -m × |i - j|

    Où m est une pente spécifique à chaque attention head.

    Exemple avec 8 heads:
      Slopes: [2^(-1), 2^(-2), 2^(-3), ..., 2^(-8)]
            = [0.5, 0.25, 0.125, ..., 0.0039]

    Avantages:
      ✅ Pas de position embeddings du tout!
      ✅ Extrapolation parfaite
      ✅ Simple à implémenter
      ✅ Pas de paramètres additionnels

    Utilisé par: BLOOM (176B), MPT (7B-65B)

    Paper: https://arxiv.org/abs/2108.12409
    """

    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Nombre d'attention heads
        """
        self.num_heads = num_heads
        self.slopes = self._get_slopes()

    def _get_slopes(self) -> np.ndarray:
        """
        Calcule les slopes pour chaque head

        Formule: m_i = 2^(-(i+1))

        Returns:
            Array de shape (num_heads,)
        """
        # Geometric sequence: 2^-1, 2^-2, 2^-3, ...
        slopes = 2.0 ** (-np.arange(1, self.num_heads + 1))

        return slopes

    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Génère la matrice de bias pour une séquence

        Args:
            seq_len: Longueur de la séquence

        Returns:
            Bias de shape (num_heads, seq_len, seq_len)

        Exemple pour seq_len=4, 1 head avec slope=0.5:
          [[0.0, -0.5, -1.0, -1.5],
           [0.0,  0.0, -0.5, -1.0],
           [0.0,  0.0,  0.0, -0.5],
           [0.0,  0.0,  0.0,  0.0]]
        """
        # Créer matrice de distances
        # positions: [0, 1, 2, ..., seq_len-1]
        positions = np.arange(seq_len)

        # Distance matrix: |i - j|
        # (seq_len, 1) - (1, seq_len) = (seq_len, seq_len)
        distances = np.abs(positions[:, None] - positions[None, :])

        # Appliquer slopes pour chaque head
        # (num_heads, 1, 1) * (1, seq_len, seq_len)
        bias = -self.slopes[:, None, None] * distances[None, :, :]

        # Masquer le futur (causal mask)
        # Token i ne peut pas attendre sur token j si j > i
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # Upper triangle
        bias = bias - causal_mask[None, :, :] * 1e9  # -inf pour futur

        return bias

    def demo(self, seq_len: int = 8):
        """Démontre le fonctionnement d'ALiBi"""
        print("\n" + "="*80)
        print("ALiBi - ATTENTION WITH LINEAR BIASES")
        print("="*80)
        print(f"Num heads: {self.num_heads}")
        print(f"Seq length: {seq_len}")

        print(f"\n📊 Slopes par head:")
        for i, slope in enumerate(self.slopes):
            print(f"  Head {i}: {slope:.4f}")

        # Générer bias pour 1 head
        bias_single = self.get_bias(seq_len)[0]  # Premier head

        print(f"\n📊 Matrice de bias (Head 0, slope={self.slopes[0]:.4f}):")
        print("     ", end="")
        for j in range(seq_len):
            print(f"  j={j}  ", end="")
        print()

        for i in range(seq_len):
            print(f"i={i}  ", end="")
            for j in range(seq_len):
                val = bias_single[i, j]
                if val < -1e8:
                    print("  -∞   ", end="")
                else:
                    print(f"{val:6.2f} ", end="")
            print()

        print(f"\n💡 Interprétation:")
        print(f"  • Valeur 0: Pas de pénalité (même position)")
        print(f"  • Valeur négative: Pénalité selon distance")
        print(f"  • -∞: Futur masqué (causal)")


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_position_encodings():
    """Compare les différents position encodings"""
    print("="*80)
    print("COMPARAISON DES POSITION ENCODINGS")
    print("="*80)

    # 1. Sinusoidal
    print("\n1️⃣  SINUSOIDAL POSITION ENCODING")
    print("-" * 80)

    sin_pe = SinusoidalPositionEncoding(d_model=512, max_len=2048)
    sin_pe.visualize_patterns()

    # Test extrapolation
    print("\n⚠️  Test d'extrapolation (10k tokens):")
    try:
        pe_10k = sin_pe.encode(10000)
        print(f"  Techniquement possible, mais qualité dégradée")
    except:
        print(f"  Erreur")

    # 2. RoPE
    print("\n\n2️⃣  RoPE - ROTARY POSITION EMBEDDING")
    print("-" * 80)

    rope = RoPE(dim=64, max_seq_len=2048)
    print(f"Dimension: {rope.dim}")
    print(f"Max seq len: {rope.max_seq_len}")
    print(f"Base: {rope.base}")
    print(f"Fréquences (premières 4): {rope.freqs[:4]}")

    # Extension de contexte
    rope.extend_context(new_max_len=8192, method="linear_scaling")

    # 3. ALiBi
    print("\n\n3️⃣  ALiBi - ATTENTION WITH LINEAR BIASES")
    print("-" * 80)

    alibi = ALiBi(num_heads=8)
    alibi.demo(seq_len=8)

    print("\n\n✅ Extrapolation ALiBi:")
    print("  • Marche nativement pour toute longueur")
    print("  • Aucun fine-tuning nécessaire")
    print("  • Utilisé pour étendre de 2k → 100k+ tokens")


def compare_context_lengths():
    """Compare l'efficacité selon longueur de contexte"""
    print("\n\n" + "="*80)
    print("EFFICACITÉ SELON LONGUEUR DE CONTEXTE")
    print("="*80)

    context_lengths = [2048, 4096, 8192, 16384, 32768, 65536, 100000]

    print(f"\n{'Context':<10} {'Attention Ops':<15} {'Memory (GB)':<15} {'Relative Cost':<15}")
    print("-" * 80)

    base_ops = context_lengths[0] ** 2
    d_model = 4096  # Dimension typique

    for ctx_len in context_lengths:
        # Opérations d'attention: O(n²)
        ops = ctx_len ** 2

        # Mémoire pour attention matrix: n × n × d_model × 2 bytes (float16)
        memory_bytes = ctx_len * ctx_len * d_model * 2
        memory_gb = memory_bytes / (1024 ** 3)

        # Coût relatif
        relative_cost = ops / base_ops

        print(f"{ctx_len:<10} {ops:>14,} {memory_gb:>14.2f} {relative_cost:>14.1f}x")

    print(f"\n💡 Observations:")
    print(f"  • 2k → 100k: 2,441x plus d'opérations")
    print(f"  • Mémoire attention: ~95 GB pour 100k contexte")
    print(f"  • Nécessite optimisations: Flash Attention, sparse attention, etc.")


if __name__ == "__main__":
    # Démo position encodings
    demo_position_encodings()

    # Comparaison coûts
    compare_context_lengths()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 1")
    print("="*80)
    print("""
1. POSITION ENCODING EST CRUCIAL
   • Transformers sont permutation-invariant
   • Position encoding permet de capturer l'ordre
   • Qualité du PE = qualité du long context

2. ÉVOLUTION DES TECHNIQUES
   Sinusoidal (2017):
     ✅ Simple, pas de params
     ❌ Extrapolation mauvaise

   Learned (GPT-2):
     ✅ Optimisé pour données
     ❌ Longueur fixe

   RoPE (2021):
     ✅ Relative positions
     ✅ Extensible avec tricks (PI, NTK)
     ✅ Utilisé: LLaMA, Mistral

   ALiBi (2022):
     ✅ Extrapolation parfaite
     ✅ Ultra-simple
     ✅ Utilisé: BLOOM, MPT

3. DÉFIS DU LONG CONTEXT
   Computational:
     • O(n²) pour attention
     • 100k tokens = 2,441x plus lent que 2k

   Mémoire:
     • Attention matrix: n² × d × 2 bytes
     • 100k tokens: ~95 GB juste pour attention

   Solutions:
     → Flash Attention (réordonnancement calculs)
     → Sparse Attention (ne pas tout attendre)
     → Sliding Window (contexte local)

PROCHAINE PARTIE: Techniques d'optimisation pour long context
    """)

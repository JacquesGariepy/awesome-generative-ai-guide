# Chapitre 8 (Suite): Techniques PEFT Avancées

## 2. Autres Méthodes PEFT

### 2.1 Adapter Layers

```python
"""
Adapter Layers: Petits modules insérés dans le modèle

Paper: "Parameter-Efficient Transfer Learning for NLP"
Authors: Houlsby et al. (Google), 2019

Principe:
- Insérer de petits bottleneck layers après chaque transformer block
- Architecture: down-project → non-linearity → up-project
- Garder le reste du modèle frozen
"""

import torch
import torch.nn as nn
from typing import Optional


class AdapterLayer(nn.Module):
    """
    Adapter layer avec bottleneck architecture

    Architecture:
        x → LayerNorm → Down-project → ReLU → Up-project → x + adapter_output
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int,
        dropout: float = 0.1,
        init_scale: float = 1e-3
    ):
        """
        Args:
            hidden_size: Dimension du modèle (e.g., 768, 4096)
            adapter_size: Taille du bottleneck (e.g., 64, 128)
                Typiquement hidden_size // 8 ou hidden_size // 16
            dropout: Dropout probability
            init_scale: Scale d'initialisation (petit pour stabilité)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.adapter_size = adapter_size

        # Layer norm avant adapter
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Down-projection: hidden_size → adapter_size
        self.down_project = nn.Linear(hidden_size, adapter_size)

        # Non-linearity
        self.activation = nn.ReLU()

        # Up-projection: adapter_size → hidden_size
        self.up_project = nn.Linear(adapter_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize avec small values
        nn.init.normal_(self.down_project.weight, std=init_scale)
        nn.init.normal_(self.up_project.weight, std=init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward avec residual connection

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            Output: (batch, seq_len, hidden_size)
        """

        # Normalize
        normalized = self.layer_norm(hidden_states)

        # Down-project
        down = self.down_project(normalized)

        # Activation
        activated = self.activation(down)

        # Dropout
        dropped = self.dropout(activated)

        # Up-project
        up = self.up_project(dropped)

        # Residual connection
        return hidden_states + up

    def get_num_params(self) -> int:
        """Retourne le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters())


class TransformerBlockWithAdapter(nn.Module):
    """
    Transformer block avec adapters insérés

    Adapters sont ajoutés:
    1. Après self-attention
    2. Après feed-forward network
    """

    def __init__(
        self,
        base_transformer_block: nn.Module,
        hidden_size: int,
        adapter_size: int
    ):
        """
        Args:
            base_transformer_block: Block transformer original (frozen)
            hidden_size: Hidden dimension
            adapter_size: Adapter bottleneck size
        """
        super().__init__()

        self.base_block = base_transformer_block
        # Freeze base block
        for param in self.base_block.parameters():
            param.requires_grad = False

        # Adapter après attention
        self.adapter_attn = AdapterLayer(hidden_size, adapter_size)

        # Adapter après FFN
        self.adapter_ffn = AdapterLayer(hidden_size, adapter_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward avec adapters

        Ordre:
        1. Self-attention → Adapter
        2. Feed-forward → Adapter
        """

        # Attention + Adapter (assuming base_block returns single output)
        attn_output = self.base_block.self_attn(hidden_states)
        attn_output = self.adapter_attn(attn_output)

        # FFN + Adapter
        ffn_output = self.base_block.mlp(attn_output)
        ffn_output = self.adapter_ffn(ffn_output)

        return ffn_output


# Exemple
if __name__ == "__main__":
    print("=== Adapter Layers ===\n")

    hidden_size = 768
    adapter_size = 64

    adapter = AdapterLayer(hidden_size, adapter_size)

    print(f"Hidden size: {hidden_size}")
    print(f"Adapter size: {adapter_size}")
    print(f"Bottleneck ratio: {hidden_size // adapter_size}x")
    print(f"Adapter params: {adapter.get_num_params():,}")
    print(f"Reduction vs full layer: {(hidden_size * hidden_size) // adapter.get_num_params()}x")

    # Test forward
    batch_size = 4
    seq_len = 128
    x = torch.randn(batch_size, seq_len, hidden_size)

    output = adapter(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Same shape (residual): {x.shape == output.shape}")
```

### 2.2 Prefix Tuning

```python
"""
Prefix Tuning: Optimiser des "virtual tokens" continus

Paper: "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
Authors: Li & Liang (Stanford), 2021

Principe:
- Ajouter des "prefix" tokens apprenables au début de chaque layer
- Ces prefixes modulent l'attention sans modifier les poids
- Comme des "soft prompts" optimisés
"""

import torch
import torch.nn as nn


class PrefixEncoder(nn.Module):
    """
    Encode les prefix parameters

    Utilise un petit MLP pour générer les prefix vectors
    à partir d'embeddings de faible dimension
    """

    def __init__(
        self,
        prefix_length: int,
        num_layers: int,
        hidden_size: int,
        prefix_hidden_size: int = 512
    ):
        """
        Args:
            prefix_length: Nombre de prefix tokens
            num_layers: Nombre de layers du modèle
            hidden_size: Hidden dimension du modèle
            prefix_hidden_size: Dimension intermédiaire pour MLP
        """
        super().__init__()

        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Embedding pour prefix (faible dimension)
        # Shape: (num_layers, prefix_length, prefix_hidden_size)
        self.prefix_embedding = nn.Parameter(
            torch.randn(num_layers, prefix_length, prefix_hidden_size)
        )

        # MLP pour projet vers hidden_size
        # Chaque layer a son propre prefix
        self.prefix_mlp = nn.Sequential(
            nn.Linear(prefix_hidden_size, prefix_hidden_size),
            nn.Tanh(),
            nn.Linear(prefix_hidden_size, hidden_size)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate prefix keys and values pour toutes les layers

        Args:
            batch_size: Batch size

        Returns:
            Prefix: (num_layers, batch_size, prefix_length, hidden_size)
        """

        # (num_layers, prefix_length, prefix_hidden_size)
        prefix_embed = self.prefix_embedding

        # Apply MLP: (num_layers, prefix_length, hidden_size)
        prefix = self.prefix_mlp(prefix_embed)

        # Expand for batch: (num_layers, batch_size, prefix_length, hidden_size)
        prefix = prefix.unsqueeze(1).expand(-1, batch_size, -1, -1)

        return prefix


class PrefixTuning(nn.Module):
    """
    Prefix Tuning complet

    Ajoute des prefix tokens virtuels qui modulent l'attention
    """

    def __init__(
        self,
        base_model: nn.Module,
        prefix_length: int = 10,
        num_layers: int = 12,
        hidden_size: int = 768
    ):
        """
        Args:
            base_model: Modèle de base (frozen)
            prefix_length: Nombre de prefix tokens
            num_layers: Nombre de layers
            hidden_size: Hidden dimension
        """
        super().__init__()

        self.base_model = base_model
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.prefix_length = prefix_length

        # Prefix encoder (seule partie trainable)
        self.prefix_encoder = PrefixEncoder(
            prefix_length=prefix_length,
            num_layers=num_layers,
            hidden_size=hidden_size
        )

    def get_prefix(self, batch_size: int) -> torch.Tensor:
        """Get prefix for batch"""
        return self.prefix_encoder(batch_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward avec prefix

        Les prefixes sont concaténés au début de la séquence
        pour chaque layer
        """

        batch_size = input_ids.size(0)

        # Generate prefixes: (num_layers, batch_size, prefix_length, hidden_size)
        prefixes = self.get_prefix(batch_size)

        # Extend attention mask pour inclure prefix
        if attention_mask is not None:
            # Add 1s for prefix positions
            prefix_attention = torch.ones(
                batch_size,
                self.prefix_length,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        # Forward through base model avec prefixes
        # (Implementation détaillée dépend du modèle spécifique)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_hidden_states=prefixes  # Hypothetical parameter
        )

        return outputs

    def get_num_trainable_params(self) -> int:
        """Retourne le nombre de paramètres entraînables"""
        return sum(
            p.numel() for p in self.prefix_encoder.parameters()
            if p.requires_grad
        )


# Exemple
if __name__ == "__main__":
    print("\n=== Prefix Tuning ===\n")

    prefix_length = 10
    num_layers = 12
    hidden_size = 768

    prefix_encoder = PrefixEncoder(
        prefix_length=prefix_length,
        num_layers=num_layers,
        hidden_size=hidden_size
    )

    trainable_params = sum(p.numel() for p in prefix_encoder.parameters())

    print(f"Prefix length: {prefix_length}")
    print(f"Num layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Trainable params: {trainable_params:,}")

    # Compare avec full model
    full_model_params = 110_000_000  # Example: BERT-base
    print(f"\nFull model params: {full_model_params:,}")
    print(f"Prefix params: {trainable_params:,} ({100*trainable_params/full_model_params:.3f}%)")

    # Test forward
    batch_size = 4
    prefixes = prefix_encoder(batch_size)
    print(f"\nPrefix shape: {prefixes.shape}")
    print(f"Expected: (num_layers={num_layers}, batch={batch_size}, prefix_len={prefix_length}, hidden={hidden_size})")
```

### 2.3 IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

```python
"""
IA3: Adapter ultra-léger

Paper: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"
Authors: Liu et al. (Google), 2022

Principe:
- Au lieu d'ajouter des poids, multiplier les activations par des vecteurs apprenables
- Extrêmement peu de paramètres (0.01% du modèle!)
- Rescale les activations dans attention et FFN
"""

import torch
import torch.nn as nn


class IA3Layer(nn.Module):
    """
    IA3 Layer: Learned vectors qui rescalent les activations

    Pour une activation h, IA3 fait: h' = h ⊙ l_v
    où ⊙ est element-wise multiply et l_v est un vecteur appris
    """

    def __init__(
        self,
        hidden_size: int,
        init_value: float = 1.0
    ):
        """
        Args:
            hidden_size: Dimension des activations
            init_value: Valeur d'initialisation (typiquement 1.0 = identité)
        """
        super().__init__()

        # Vecteur de scaling (un seul paramètre par dimension!)
        self.scaling_vector = nn.Parameter(
            torch.ones(hidden_size) * init_value
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply element-wise scaling

        Args:
            hidden_states: (..., hidden_size)

        Returns:
            Scaled activations: (..., hidden_size)
        """

        return hidden_states * self.scaling_vector


class TransformerWithIA3(nn.Module):
    """
    Transformer avec IA3 adapters

    IA3 vectors sont appliqués à:
    1. Keys et Values dans self-attention
    2. FFN intermediate activations
    """

    def __init__(
        self,
        base_transformer: nn.Module,
        hidden_size: int,
        ffn_hidden_size: int
    ):
        """
        Args:
            base_transformer: Transformer block (frozen)
            hidden_size: Hidden dimension
            ffn_hidden_size: FFN intermediate dimension
        """
        super().__init__()

        self.base_transformer = base_transformer
        for param in self.base_transformer.parameters():
            param.requires_grad = False

        # IA3 pour attention keys
        self.ia3_k = IA3Layer(hidden_size)

        # IA3 pour attention values
        self.ia3_v = IA3Layer(hidden_size)

        # IA3 pour FFN intermediate
        self.ia3_ffn = IA3Layer(ffn_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward avec IA3 scaling

        Note: Pseudocode, l'implémentation exacte dépend du modèle
        """

        # Dans self-attention:
        # Q = Wq @ x
        # K = (Wk @ x) ⊙ l_k  ← IA3 scaling
        # V = (Wv @ x) ⊙ l_v  ← IA3 scaling

        # Dans FFN:
        # h = activation(W1 @ x) ⊙ l_ffn  ← IA3 scaling
        # y = W2 @ h

        # Implementation détaillée omise car dépend de l'architecture
        pass

    def get_num_trainable_params(self) -> int:
        """Nombre de paramètres entraînables"""
        return sum(
            p.numel() for p in [self.ia3_k, self.ia3_v, self.ia3_ffn]
            for p in p.parameters()
        )


# Exemple
if __name__ == "__main__":
    print("\n=== IA3 ===\n")

    hidden_size = 4096
    ffn_hidden_size = 16384  # Typiquement 4x hidden_size

    # Nombre de layers
    num_layers = 32  # Example: Llama 2 7B

    # IA3 params par layer
    params_per_layer = 2 * hidden_size + ffn_hidden_size  # K, V, FFN

    total_ia3_params = params_per_layer * num_layers

    print(f"Hidden size: {hidden_size}")
    print(f"FFN hidden: {ffn_hidden_size}")
    print(f"Num layers: {num_layers}")
    print(f"\nIA3 params per layer: {params_per_layer:,}")
    print(f"Total IA3 params: {total_ia3_params:,}")

    # Compare avec modèle complet
    full_model_params = 7_000_000_000  # Llama 2 7B
    print(f"\nFull model: {full_model_params:,}")
    print(f"IA3 trainable: {total_ia3_params:,} ({100*total_ia3_params/full_model_params:.4f}%)")

    print(f"\nIA3 est ~10,000x plus léger que LoRA!")
```

## 2.4 Comparaison Complète des Méthodes PEFT

```python
"""
Tableau de comparaison exhaustif de toutes les méthodes PEFT
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PEFTMethod:
    """Caractéristiques d'une méthode PEFT"""
    name: str
    trainable_params_pct: str
    memory_requirement: str
    training_speed: str
    inference_latency: str
    performance_vs_full: str
    ease_of_use: str
    modularity: str
    best_for: List[str]
    limitations: List[str]


class ComprehensivePEFTComparison:
    """Comparaison exhaustive de toutes les méthodes"""

    @staticmethod
    def get_all_methods() -> Dict[str, PEFTMethod]:
        """Retourne toutes les méthodes PEFT"""

        return {
            "LoRA": PEFTMethod(
                name="LoRA",
                trainable_params_pct="0.1-1%",
                memory_requirement="Moyen (1.5-2x base)",
                training_speed="Rapide (2-3x vs full FT)",
                inference_latency="Bas (merge possible)",
                performance_vs_full="95-99%",
                ease_of_use="⭐⭐⭐⭐⭐ Excellent",
                modularity="⭐⭐⭐⭐⭐ Excellent",
                best_for=[
                    "Usage général",
                    "Multi-task (swap adapters)",
                    "Datasets moyens à grands",
                    "Production deployment"
                ],
                limitations=[
                    "Nécessite tuning de rank/alpha",
                    "Légèrement moins performant que full FT"
                ]
            ),

            "QLoRA": PEFTMethod(
                name="QLoRA",
                trainable_params_pct="0.1-1%",
                memory_requirement="Très bas (0.4-0.6x base) 🏆",
                training_speed="Moyen (légèrement plus lent que LoRA)",
                inference_latency="Bas (merge possible)",
                performance_vs_full="95-99%",
                ease_of_use="⭐⭐⭐⭐ Très bon",
                modularity="⭐⭐⭐⭐⭐ Excellent",
                best_for=[
                    "GPUs limités (8-24GB)",
                    "Très grands modèles (70B+)",
                    "Consumer hardware",
                    "Budget limité"
                ],
                limitations=[
                    "Nécessite bitsandbytes",
                    "Légèrement plus lent que LoRA FP16",
                    "Quantization peut affecter qualité"
                ]
            ),

            "Adapters": PEFTMethod(
                name="Adapter Layers",
                trainable_params_pct="1-3%",
                memory_requirement="Moyen (2-2.5x base)",
                training_speed="Moyen",
                inference_latency="Moyen (forward pass extra layers)",
                performance_vs_full="90-95%",
                ease_of_use="⭐⭐⭐⭐ Très bon",
                modularity="⭐⭐⭐⭐⭐ Excellent",
                best_for=[
                    "Multi-task learning",
                    "Modularité maximale",
                    "Interpretabilité"
                ],
                limitations=[
                    "Plus de params que LoRA",
                    "Latence d'inférence plus haute",
                    "Performance légèrement inférieure"
                ]
            ),

            "Prefix Tuning": PEFTMethod(
                name="Prefix Tuning",
                trainable_params_pct="0.01-0.1%",
                memory_requirement="Bas (1.2-1.5x base)",
                training_speed="Rapide",
                inference_latency="Bas",
                performance_vs_full="85-95% (variable)",
                ease_of_use="⭐⭐⭐ Moyen",
                modularity="⭐⭐⭐⭐ Bon",
                best_for=[
                    "Generation tasks",
                    "Prompt optimization",
                    "Très peu de ressources"
                ],
                limitations=[
                    "Performance variable selon tâche",
                    "Moins bon pour NLU complexe",
                    "Difficile à debugger"
                ]
            ),

            "P-Tuning v2": PEFTMethod(
                name="P-Tuning v2",
                trainable_params_pct="0.1-0.5%",
                memory_requirement="Bas (1.5-2x base)",
                training_speed="Rapide",
                inference_latency="Bas",
                performance_vs_full="90-95%",
                ease_of_use="⭐⭐⭐ Moyen",
                modularity="⭐⭐⭐⭐ Bon",
                best_for=[
                    "NLU tasks",
                    "Question answering",
                    "Classification"
                ],
                limitations=[
                    "Moins bon que LoRA en général",
                    "Spécifique à certaines tâches"
                ]
            ),

            "IA3": PEFTMethod(
                name="IA3",
                trainable_params_pct="0.001-0.01% 🏆",
                memory_requirement="Très bas (1.1-1.2x base)",
                training_speed="Très rapide 🏆",
                inference_latency="Très bas 🏆",
                performance_vs_full="85-92%",
                ease_of_use="⭐⭐⭐⭐ Très bon",
                modularity="⭐⭐⭐⭐ Bon",
                best_for=[
                    "Ressources extrêmement limitées",
                    "Many-task scenarios",
                    "Rapid prototyping"
                ],
                limitations=[
                    "Performance inférieure aux autres",
                    "Moins flexible que LoRA",
                    "Nouveau, moins testé"
                ]
            ),

            "Full Fine-Tuning": PEFTMethod(
                name="Full Fine-Tuning",
                trainable_params_pct="100%",
                memory_requirement="Très élevé (6x base FP32, 3x FP16)",
                training_speed="Baseline (1x)",
                inference_latency="Bas (no overhead)",
                performance_vs_full="100% (baseline) 🏆",
                ease_of_use="⭐⭐⭐⭐⭐ Excellent",
                modularity="⭐ Mauvais",
                best_for=[
                    "Datasets très larges",
                    "Ressources illimitées",
                    "Performance maximale absolue"
                ],
                limitations=[
                    "Très coûteux",
                    "Lent",
                    "Risque d'overfitting élevé",
                    "Nécessite GPUs haut de gamme"
                ]
            )
        }

    @staticmethod
    def print_comparison_table():
        """Affiche un tableau de comparaison"""

        methods = ComprehensivePEFTComparison.get_all_methods()

        print("="*80)
        print("COMPARAISON COMPLÈTE DES MÉTHODES PEFT")
        print("="*80)
        print()

        for name, method in methods.items():
            print(f"### {method.name}")
            print(f"Paramètres entraînables: {method.trainable_params_pct}")
            print(f"Mémoire: {method.memory_requirement}")
            print(f"Vitesse training: {method.training_speed}")
            print(f"Latence inférence: {method.inference_latency}")
            print(f"Performance vs Full FT: {method.performance_vs_full}")
            print(f"Facilité d'utilisation: {method.ease_of_use}")
            print(f"Modularité: {method.modularity}")
            print(f"\nIdéal pour:")
            for use_case in method.best_for:
                print(f"  ✅ {use_case}")
            print(f"\nLimitations:")
            for limitation in method.limitations:
                print(f"  ⚠️  {limitation}")
            print()
            print("-"*80)
            print()


# Exemple
if __name__ == "__main__":
    ComprehensivePEFTComparison.print_comparison_table()

    print("\n" + "="*80)
    print("RECOMMANDATIONS")
    print("="*80)
    print()

    recommendations = {
        "GPU 8-16GB, modèle 7B": "→ QLoRA (rank=16-32)",
        "GPU 24GB, modèle 13B": "→ LoRA FP16 (rank=16-64)",
        "GPU 40GB+, modèle 70B": "→ QLoRA (rank=32-64)",
        "Multi-task deployment": "→ LoRA (swap adapters facilement)",
        "Budget très limité": "→ QLoRA ou IA3",
        "Performance maximale": "→ Full Fine-Tuning (si ressources)",
        "Rapid prototyping": "→ LoRA (balance parfait)",
        "Generation tasks": "→ LoRA ou Prefix Tuning",
        "NLU tasks": "→ LoRA ou P-Tuning v2"
    }

    for scenario, recommendation in recommendations.items():
        print(f"{scenario:.<40} {recommendation}")
```

*[Suite avec QLoRA et quantization dans la partie 3...]*

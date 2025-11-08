# Chapitre 14 (Fin): Projets, Best Practices et Conclusion

## Projet 2: DPO Training Simplifié

```python
"""
Projet Pratique 2: DPO Training avec TRL

DPO est plus simple que RLHF - pas de reward model, pas de PPO
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DPOProjectConfig:
    """Configuration pour projet DPO"""

    base_model: str = "meta-llama/Llama-2-7b-hf"
    sft_model_path: str = "./models/sft-llama-2-7b"
    dpo_model_path: str = "./models/dpo-llama-2-7b"

    dataset: str = "Anthropic/hh-rlhf"
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_epochs: int = 3
    batch_size: int = 4


class DPOProject:
    """Pipeline DPO complet"""

    def __init__(self, config: DPOProjectConfig):
        self.config = config

    def step1_sft(self):
        """Étape 1: SFT (identique à RLHF)"""

        print("="*60)
        print("ÉTAPE 1: Supervised Fine-Tuning")
        print("="*60 + "\n")

        code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Prepare dataset
dataset = load_dataset("OpenAssistant/oasst1")

# Train
training_args = TrainingArguments(
    output_dir="./sft-llama-2-7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./models/sft-llama-2-7b")
        '''

        print(code)
        print("\n✅ SFT Model prêt\n")

    def step2_dpo_training(self):
        """Étape 2: DPO Training directement"""

        print("="*60)
        print("ÉTAPE 2: DPO Training (PAS de reward model!)")
        print("="*60 + "\n")

        code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 1. Load SFT model (sera le policy ET le reference)
model = AutoModelForCausalLM.from_pretrained("./models/sft-llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("./models/sft-llama-2-7b")

# 2. Load preference dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# Dataset format:
# {
#   "prompt": "...",
#   "chosen": "...",    # Réponse préférée
#   "rejected": "..."   # Réponse non-préférée
# }

# 3. DPO Configuration
dpo_config = DPOConfig(
    output_dir="./models/dpo-llama-2-7b",
    beta=0.1,  # Temperature parameter
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    max_length=512,
    max_prompt_length=256,
)

# 4. Create DPO Trainer
# Note: TRL gère automatiquement la création du reference model
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # TRL créera une copie du model comme ref
    args=dpo_config,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
)

# 5. Train!
dpo_trainer.train()

# 6. Save
dpo_trainer.save_model("./models/dpo-llama-2-7b")
        '''

        print(code)
        print("\n✅ DPO Model entraîné - C'est TOUT!")
        print("Pas besoin de reward model ni de PPO!\n")

    def step3_comparison(self):
        """Étape 3: Comparer RLHF vs DPO"""

        print("="*60)
        print("ÉTAPE 3: Comparaison RLHF vs DPO")
        print("="*60 + "\n")

        comparison = '''
=== Complexité ===

RLHF Pipeline:
  1. SFT Training         [~8 heures sur 1x A100]
  2. Reward Model Training [~4 heures]
  3. PPO Training          [~16 heures]
  ─────────────────────────────────────────
  TOTAL: ~28 heures + complexité PPO

DPO Pipeline:
  1. SFT Training         [~8 heures sur 1x A100]
  2. DPO Training         [~8 heures]
  ─────────────────────────────────────────
  TOTAL: ~16 heures - 40% plus rapide!

=== Code ===

RLHF: ~500 lignes (reward model + PPO implementation)
DPO:  ~100 lignes (juste DPO trainer)

=== Résultats ===

Performance (sur benchmarks):
  RLHF: ★★★★★ (5/5)
  DPO:  ★★★★☆ (4/5) - Légèrement inférieur mais très proche

Stabilité:
  RLHF: ★★☆☆☆ (instable, sensible aux hyperparams)
  DPO:  ★★★★☆ (beaucoup plus stable)

=== Recommandation ===

Commencer par DPO:
  ✅ Plus simple
  ✅ Plus rapide
  ✅ Plus stable
  ✅ Résultats excellents (90-95% de RLHF)

Passer à RLHF si:
  • Budget compute important
  • Besoin des derniers 5-10% de performance
  • Équipe avec expertise RL
  • Cas d'usage très spécifique nécessitant flexibilité reward
        '''

        print(comparison)

    def run_pipeline(self):
        """Exécute le pipeline DPO"""

        print("\n" + "="*60)
        print("PIPELINE DPO COMPLET")
        print("="*60 + "\n")

        self.step1_sft()
        self.step2_dpo_training()
        self.step3_comparison()

        print("\n" + "="*60)
        print("✅ PIPELINE DPO TERMINÉ")
        print("="*60 + "\n")


# Exemple d'utilisation
if __name__ == "__main__":
    config = DPOProjectConfig(
        base_model="meta-llama/Llama-2-7b-hf",
        beta=0.1,
        num_epochs=3
    )

    project = DPOProject(config)
    project.run_pipeline()
```

---

## Best Practices et Conseils

### 5.1 Préparation des Données

```python
"""
Best practices pour la préparation des données RLHF/DPO
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PreferenceQuality:
    """Critères de qualité pour les préférences"""

    agreement_threshold: float = 0.7
    """Au moins 70% d'annotateurs doivent être d'accord"""

    min_margin: float = 0.3
    """Marge minimale entre chosen et rejected"""

    max_length_ratio: float = 2.0
    """Ratio maximal de longueur chosen/rejected"""


class DataPreparationBestPractices:
    """Best practices pour la préparation des données"""

    @staticmethod
    def guidelines() -> Dict[str, List[str]]:
        """Guidelines pour la collecte de préférences"""

        return {
            "Collection de Préférences": [
                "Utiliser plusieurs annotateurs (3-5) par paire",
                "Mesurer l'inter-annotator agreement (Kappa > 0.6)",
                "Fournir des guidelines claires et cohérentes",
                "Inclure des exemples de calibration",
                "Vérifier les biais démographiques des annotateurs"
            ],

            "Qualité des Paires": [
                "Éviter les paires trop similaires (hard to distinguish)",
                "Éviter les paires trop différentes (trivial)",
                "Sweet spot: différences claires mais subtiles",
                "Vérifier la diversité des topics et styles",
                "Inclure des edge cases"
            ],

            "Taille du Dataset": [
                "Minimum ~10k paires pour DPO",
                "Minimum ~50k paires pour RLHF (reward model)",
                "Plus de données = modèle plus robuste",
                "Mais qualité > quantité!",
                "Augmenter graduellement si besoin"
            ],

            "Équilibre": [
                "Balancer les catégories (helpful, harmless, honest)",
                "Balancer les longueurs de réponses",
                "Balancer les difficultés",
                "Éviter les biais systématiques"
            ],

            "Validation": [
                "Séparer train/validation/test (80/10/10)",
                "Test set doit être representative",
                "Monitorer overfitting sur validation",
                "Test set pour evaluation finale uniquement"
            ]
        }

    @staticmethod
    def data_cleaning_pipeline() -> str:
        """Pipeline de nettoyage de données"""

        return """
=== Pipeline de Nettoyage ===

1. Déduplication
   • Supprimer les paires identiques
   • Supprimer les prompts dupliqués avec réponses différentes
   • Hash-based deduplication

2. Filtrage de Qualité
   • Supprimer paires trop courtes (< 10 tokens)
   • Supprimer paires trop longues (> 2048 tokens)
   • Supprimer si ratio longueur > 3x
   • Filtrer par score de cohérence (perplexité)

3. Filtrage de Contenu
   • Supprimer contenu toxique/offensant
   • Supprimer PII (emails, numéros, etc.)
   • Filtrer selon content policy
   • Vérifier avec modérateur de contenu

4. Vérification de Préférences
   • Vérifier inter-annotator agreement
   • Supprimer si agreement < threshold
   • Résoudre conflits avec annotateur expert
   • Documenter edge cases

5. Augmentation (Optionnel)
   • Paraphraser prompts
   • Générer variations synthétiques
   • Back-translation
   • Mais attention: peut introduire du bruit

6. Final Checks
   • Statistiques du dataset (longueurs, topics, etc.)
   • Distribution des scores
   • Vérifier équilibre train/val/test
   • Documentation complète
        """

    @staticmethod
    def example_filtering_code() -> str:
        """Code exemple pour filtrage"""

        return '''
def filter_preference_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    agreement_score: float,
    config: PreferenceQuality
) -> bool:
    """
    Filtre une paire de préférences selon critères de qualité

    Returns:
        True si la paire est acceptable, False sinon
    """

    # 1. Agreement threshold
    if agreement_score < config.agreement_threshold:
        return False

    # 2. Length checks
    if len(chosen.split()) < 10 or len(rejected.split()) < 10:
        return False

    chosen_len = len(chosen.split())
    rejected_len = len(rejected.split())
    length_ratio = max(chosen_len, rejected_len) / min(chosen_len, rejected_len)

    if length_ratio > config.max_length_ratio:
        return False

    # 3. Content checks
    if contains_pii(chosen) or contains_pii(rejected):
        return False

    if is_toxic(chosen) or is_toxic(rejected):
        return False

    # 4. Similarity check (trop similaires = pas utile)
    similarity = compute_similarity(chosen, rejected)
    if similarity > 0.9:
        return False

    return True


def compute_preference_margin(
    chosen_score: float,
    rejected_score: float
) -> float:
    """
    Calcule la marge entre chosen et rejected

    Une marge trop faible indique que la préférence n'est pas claire
    """
    return abs(chosen_score - rejected_score)


# Application sur dataset
def clean_preference_dataset(dataset, config):
    """Nettoie un dataset de préférences"""

    cleaned = []

    for example in dataset:
        # Filter
        if filter_preference_pair(
            example['prompt'],
            example['chosen'],
            example['rejected'],
            example['agreement'],
            config
        ):
            cleaned.append(example)

    print(f"Dataset size: {len(dataset)} → {len(cleaned)}")
    print(f"Kept: {len(cleaned)/len(dataset):.1%}")

    return cleaned
        '''


# Exemple d'utilisation
if __name__ == "__main__":
    bp = DataPreparationBestPractices()

    print("=== Best Practices pour Données RLHF/DPO ===\n")

    guidelines = bp.guidelines()
    for category, points in guidelines.items():
        print(f"\n## {category}")
        for point in points:
            print(f"  • {point}")

    print("\n" + "="*60 + "\n")
    print(bp.data_cleaning_pipeline())
```

### 5.2 Monitoring et Debugging

```python
"""
Best practices pour monitoring et debugging RLHF/DPO
"""

from typing import Dict, List
import numpy as np


class RLHFMonitoring:
    """Métriques à monitorer pendant RLHF training"""

    @staticmethod
    def key_metrics() -> Dict[str, str]:
        """Métriques clés à surveiller"""

        return {
            # Reward metrics
            "mean_reward": "Reward moyen par batch - doit augmenter",
            "reward_std": "Écart-type des rewards - stabilité",
            "reward_distribution": "Distribution des rewards - vérifier shift",

            # KL divergence
            "mean_kl": "KL moyen π vs π_ref - doit rester < target_kl",
            "max_kl": "KL max - attention si > 2*target_kl",
            "kl_coef": "Coefficient KL adaptif - surveiller ajustements",

            # Policy metrics
            "policy_loss": "Loss PPO - doit décroître",
            "value_loss": "Loss value function - doit décroître",
            "entropy": "Entropie de la policy - exploration",
            "clipfrac": "Fraction clippée - si > 0.3, reduire cliprange",

            # Generation quality
            "response_length": "Longueur moyenne des réponses",
            "unique_tokens": "Diversité du vocabulaire",
            "repetition_rate": "Taux de répétition - attention si > 0.1"
        }

    @staticmethod
    def warning_signs() -> Dict[str, str]:
        """Signes d'alerte pendant training"""

        return {
            "Reward Hacking": {
                "symptom": "Reward augmente mais qualité diminue",
                "check": "Évaluation humaine diverge du reward model",
                "solution": "Augmenter β (KL coef), diversifier reward model"
            },

            "Mode Collapse": {
                "symptom": "Réponses toutes similaires, faible diversité",
                "check": "Entropie très basse, repetition_rate élevé",
                "solution": "Augmenter entropy bonus, réduire training"
            },

            "KL Explosion": {
                "symptom": "KL divergence augmente rapidement",
                "check": "mean_kl >> target_kl",
                "solution": "Augmenter kl_coef, réduire learning rate"
            },

            "Value Function Collapse": {
                "symptom": "Value loss stagne ou augmente",
                "check": "value_loss ne diminue pas",
                "solution": "Augmenter vf_coef, vérifier advantage computation"
            },

            "Clipfrac trop élevé": {
                "symptom": "Beaucoup de clipping (> 30%)",
                "check": "clipfrac > 0.3",
                "solution": "Réduire learning rate ou reduire cliprange"
            }
        }

    @staticmethod
    def debugging_checklist() -> str:
        """Checklist pour debugging"""

        return """
=== RLHF Debugging Checklist ===

Si le training ne fonctionne pas:

□ Data Quality
  • Vérifier format du dataset
  • Vérifier qualité des préférences
  • Statistiques du dataset (longueurs, distribution)

□ Reward Model
  • Accuracy > 60% sur validation?
  • Reward distribution raisonnable?
  • Test sur exemples connus

□ PPO Configuration
  • Learning rate pas trop élevé? (< 1e-5)
  • KL coefficient raisonnable? (0.1-0.5)
  • Batch size suffisant? (>= 16)
  • Cliprange approprié? (0.1-0.3)

□ Generation Quality
  • Générer exemples et inspecter manuellement
  • Comparer avec SFT baseline
  • Vérifier répétitions, cohérence

□ Monitoring
  • Tous les metrics loggés?
  • Visualisation des courbes
  • Checkpoints sauvegardés?

□ Resources
  • Mémoire GPU suffisante?
  • Batch size ajusté selon GPU?
  • Gradient accumulation si needed?
        """


# Exemple d'utilisation
if __name__ == "__main__":
    monitoring = RLHFMonitoring()

    print("=== Métriques Clés RLHF ===\n")
    for metric, description in monitoring.key_metrics().items():
        print(f"{metric:20s}: {description}")

    print("\n" + "="*60 + "\n")
    print("=== Signes d'Alerte ===\n")

    for issue, details in monitoring.warning_signs().items():
        print(f"\n## {issue}")
        print(f"Symptôme: {details['symptom']}")
        print(f"Vérifier: {details['check']}")
        print(f"Solution: {details['solution']}")

    print("\n" + "="*60 + "\n")
    print(monitoring.debugging_checklist())
```

---

## Conclusion

### Résumé du Chapitre

Ce chapitre a couvert en profondeur le **Reinforcement Learning from Human Feedback (RLHF)** et les techniques associées pour aligner les LLMs avec les préférences humaines.

#### Ce que vous avez appris:

1. **Foundations RLHF**
   - Pipeline complet en 3 étapes (SFT → Reward Model → PPO)
   - Motivation et théorie
   - Composants et interactions

2. **Reward Modeling**
   - Architecture et entraînement
   - Pairwise ranking loss
   - Évaluation et métriques

3. **PPO (Proximal Policy Optimization)**
   - Théorie et formulation mathématique
   - Implementation complète from scratch
   - GAE (Generalized Advantage Estimation)
   - KL divergence penalty

4. **DPO (Direct Preference Optimization)**
   - Alternative plus simple à RLHF
   - Pas de reward model, pas de PPO
   - Résultats comparables avec moins de complexité

5. **Projets Pratiques**
   - Pipeline RLHF complet avec TRL
   - Pipeline DPO simplifié
   - Best practices et debugging

#### Comparaison Finale

| Critère | RLHF | DPO | Recommandation |
|---------|------|-----|----------------|
| **Complexité** | ★★★★★ | ★★★☆☆ | DPO pour débuter |
| **Performances** | ★★★★★ | ★★★★☆ | RLHF si budget OK |
| **Stabilité** | ★★☆☆☆ | ★★★★☆ | DPO plus stable |
| **Coût Compute** | ★★★★★ | ★★★☆☆ | DPO 40% moins cher |
| **Temps Training** | ~28h | ~16h | DPO plus rapide |

### Prochaines Étapes

1. **Pratiquer**: Implémenter les projets pratiques
2. **Expérimenter**: Tester différents hyperparamètres
3. **Évaluer**: Comparer RLHF vs DPO sur vos données
4. **Optimiser**: Fine-tuner pour votre use case spécifique
5. **Deployer**: Mettre en production avec guardrails (Ch. 16)

### Ressources Additionnelles

#### Papers Fondamentaux

1. **RLHF**
   - "Training language models to follow instructions with human feedback" (InstructGPT, OpenAI, 2022)
   - "Learning to summarize from human feedback" (OpenAI, 2020)

2. **PPO**
   - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

3. **DPO**
   - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)

4. **Constitutional AI**
   - "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)

#### Librairies et Outils

1. **TRL (Transformer Reinforcement Learning)**
   - https://github.com/huggingface/trl
   - La référence pour RLHF/DPO avec HuggingFace

2. **DeepSpeed-Chat**
   - https://github.com/microsoft/DeepSpeed
   - RLHF optimisé pour grandes échelles

3. **OpenRLHF**
   - https://github.com/OpenLLMAI/OpenRLHF
   - Implementation open-source de RLHF

#### Datasets

1. **Anthropic HH-RLHF**
   - ~170k comparaisons humaines
   - Helpful et Harmless

2. **OpenAssistant**
   - ~160k conversations
   - Multilingue

3. **Stanford Human Preferences**
   - ~330k comparaisons
   - Diverses catégories

#### Blogs et Tutorials

1. **HuggingFace Blog**
   - "Illustrating Reinforcement Learning from Human Feedback"
   - "Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU"

2. **OpenAI Blog**
   - "InstructGPT: Training language models to follow instructions"

3. **Anthropic Blog**
   - "Constitutional AI: Harmlessness from AI Feedback"

### Points Clés à Retenir

1. **RLHF transforme les LLMs** de simples compléteurs de texte en assistants utiles et alignés
2. **PPO est complexe** mais extrêmement efficace quand bien configuré
3. **DPO simplifie** le processus tout en maintenant d'excellentes performances
4. **La qualité des données** de préférences est CRITIQUE
5. **Le monitoring** est essentiel pour éviter reward hacking et mode collapse
6. **KL divergence penalty** empêche le modèle de trop s'éloigner de la référence
7. **Commencer simple** (DPO) puis complexifier si nécessaire (RLHF)
8. **Évaluation humaine** reste indispensable malgré les métriques automatiques

### Exercices Pratiques

1. **Exercice 1**: Implémenter un reward model simple sur un petit dataset
2. **Exercice 2**: Entraîner un modèle avec DPO sur Anthropic HH-RLHF
3. **Exercice 3**: Comparer SFT, DPO et RLHF sur des métriques objectives
4. **Exercice 4**: Identifier et corriger des cas de reward hacking
5. **Exercice 5**: Créer un dashboard de monitoring pour RLHF training

---

**Fin du Chapitre 14**

Dans le prochain chapitre, nous explorerons les **APIs et Services** pour déployer vos LLMs alignés en production.

**Rappel**: Le Chapitre 16 (Sécurité et Éthique) couvre les guardrails essentiels à mettre en place AVANT le déploiement!

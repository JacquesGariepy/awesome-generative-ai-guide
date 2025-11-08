# Chapitre 7: Fine-tuning - Techniques et Pratiques

## Table des Matières
1. [Introduction](#introduction)
2. [Pourquoi Fine-tuner?](#pourquoi-fine-tuner)
3. [Types de Fine-tuning](#types-de-fine-tuning)
4. [Préparation des Données](#préparation-des-données)
5. [Hyperparamètres et Optimisation](#hyperparamètres-et-optimisation)
6. [Prévention de l'Overfitting](#prévention-de-loverfitting)
7. [Évaluation et Métriques](#évaluation-et-métriques)
8. [Projets Pratiques](#projets-pratiques)
9. [Conclusion](#conclusion)

---

## Introduction

Le **fine-tuning** (affinage) est l'une des techniques les plus puissantes pour adapter un LLM pré-entraîné à votre cas d'usage spécifique. Ce chapitre couvre toutes les techniques de fine-tuning, de la théorie à la pratique.

### Objectifs d'Apprentissage

À la fin de ce chapitre, vous serez capable de:

1. Comprendre quand fine-tuner vs prompt engineering
2. Préparer des datasets de qualité pour le fine-tuning
3. Choisir les bons hyperparamètres
4. Implémenter full fine-tuning avec HuggingFace
5. Prévenir l'overfitting et optimiser la généralisation
6. Évaluer rigoureusement vos modèles fine-tunés
7. Déployer un modèle fine-tuné en production

### Contexte

Les LLMs pré-entraînés sont excellents pour des tâches générales, mais souvent sous-optimaux pour:

- ✅ Votre domaine spécifique (médical, légal, finance, etc.)
- ✅ Votre ton et style de communication
- ✅ Vos formats de sortie spécifiques
- ✅ Vos données propriétaires

Le fine-tuning permet d'adapter le modèle tout en conservant ses capacités générales.

---

## Pourquoi Fine-tuner?

### 1.1 Fine-tuning vs Autres Approches

```python
"""
Comparaison des approches d'adaptation de LLMs
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class AdaptationApproach(Enum):
    """Différentes approches pour adapter un LLM"""
    PROMPT_ENGINEERING = "prompt_engineering"
    FEW_SHOT_LEARNING = "few_shot_learning"
    RAG = "rag"
    FINE_TUNING = "fine_tuning"
    PRETRAINING = "pretraining"


@dataclass
class ApproachCharacteristics:
    """Caractéristiques d'une approche d'adaptation"""
    name: str
    approach: AdaptationApproach

    # Coûts
    data_requirement: str
    compute_cost: int  # 1-5
    time_cost: int  # 1-5
    complexity: int  # 1-5

    # Bénéfices
    performance: int  # 1-5
    customization: int  # 1-5
    domain_adaptation: int  # 1-5

    # Cas d'usage
    best_for: List[str]
    limitations: List[str]


class AdaptationComparison:
    """Comparaison des approches d'adaptation"""

    def __init__(self):
        self.approaches = self._initialize_approaches()

    def _initialize_approaches(self) -> Dict[str, ApproachCharacteristics]:
        """Initialise les approches"""

        return {
            "prompt_engineering": ApproachCharacteristics(
                name="Prompt Engineering",
                approach=AdaptationApproach.PROMPT_ENGINEERING,

                # Coûts
                data_requirement="Aucune à quelques exemples",
                compute_cost=1,
                time_cost=1,
                complexity=1,

                # Bénéfices
                performance=2,
                customization=2,
                domain_adaptation=1,

                # Cas d'usage
                best_for=[
                    "Tâches simples et générales",
                    "Prototypage rapide",
                    "Zéro budget compute",
                    "Tâches one-shot"
                ],
                limitations=[
                    "Performance limitée",
                    "Difficile pour tâches complexes",
                    "Pas de mémorisation de domaine",
                    "Dépend de la qualité du prompt"
                ]
            ),

            "few_shot": ApproachCharacteristics(
                name="Few-Shot Learning (In-Context)",
                approach=AdaptationApproach.FEW_SHOT_LEARNING,

                data_requirement="5-50 exemples",
                compute_cost=2,
                time_cost=1,
                complexity=2,

                performance=3,
                customization=3,
                domain_adaptation=2,

                best_for=[
                    "Tâches avec peu d'exemples disponibles",
                    "Format de sortie spécifique",
                    "Prototypage",
                    "Modèles avec grand contexte"
                ],
                limitations=[
                    "Limité par taille du contexte",
                    "Coûteux en tokens (latence + $)",
                    "Pas de vraie adaptation du modèle",
                    "Difficile de scaler"
                ]
            ),

            "rag": ApproachCharacteristics(
                name="RAG (Retrieval-Augmented Generation)",
                approach=AdaptationApproach.RAG,

                data_requirement="Base de connaissances",
                compute_cost=3,
                time_cost=2,
                complexity=3,

                performance=4,
                customization=3,
                domain_adaptation=4,

                best_for=[
                    "Knowledge-intensive tasks",
                    "Informations changeantes fréquemment",
                    "Besoin de sources/citations",
                    "Grandes bases de connaissances"
                ],
                limitations=[
                    "Nécessite vector DB et embedding",
                    "Latence de retrieval",
                    "Qualité dépend du retrieval",
                    "Ne change pas le comportement du modèle"
                ]
            ),

            "fine_tuning": ApproachCharacteristics(
                name="Fine-Tuning",
                approach=AdaptationApproach.FINE_TUNING,

                data_requirement="100-10,000+ exemples",
                compute_cost=4,
                time_cost=3,
                complexity=4,

                performance=5,
                customization=5,
                domain_adaptation=5,

                best_for=[
                    "Domaine très spécifique",
                    "Ton/style particulier",
                    "Performance critique",
                    "Données propriétaires importantes"
                ],
                limitations=[
                    "Nécessite données de qualité",
                    "Coût compute significatif",
                    "Risque d'overfitting",
                    "Expertise ML requise"
                ]
            ),

            "pretraining": ApproachCharacteristics(
                name="Pre-training from Scratch",
                approach=AdaptationApproach.PRETRAINING,

                data_requirement="Millions-Milliards exemples",
                compute_cost=5,
                time_cost=5,
                complexity=5,

                performance=5,
                customization=5,
                domain_adaptation=5,

                best_for=[
                    "Nouveau domaine/langue inexploré",
                    "Besoin de contrôle total",
                    "Budget compute massif",
                    "Recherche avancée"
                ],
                limitations=[
                    "Extrêmement coûteux ($$$$)",
                    "Temps très long (semaines/mois)",
                    "Expertise poussée requise",
                    "Risque élevé d'échec"
                ]
            )
        }

    def recommend(
        self,
        data_size: int,
        budget: str,
        performance_need: str,
        expertise: str
    ) -> str:
        """
        Recommande une approche basée sur le contexte

        Args:
            data_size: Nombre d'exemples disponibles
            budget: "low", "medium", "high"
            performance_need: "basic", "good", "excellent"
            expertise: "beginner", "intermediate", "advanced"

        Returns:
            Approche recommandée
        """

        # Arbre de décision simplifié
        if data_size < 10:
            return "prompt_engineering ou few_shot"

        elif data_size < 100:
            if budget == "low":
                return "few_shot"
            else:
                return "rag ou fine_tuning (si données de qualité)"

        elif data_size < 1000:
            if performance_need == "excellent":
                return "fine_tuning"
            else:
                return "rag ou fine_tuning"

        else:  # data_size >= 1000
            if budget == "high" and expertise == "advanced":
                return "fine_tuning (fortement recommandé)"
            elif budget == "medium":
                return "fine_tuning ou rag"
            else:
                return "rag"

    def generate_comparison_table(self) -> str:
        """Génère une table comparative"""

        table = """
=== Tableau Comparatif des Approches ===

┌─────────────────┬──────┬──────┬──────┬───────┬──────┬────────┐
│ Approche        │ Data │ Cost │ Time │ Perf  │Custom│ Domain │
├─────────────────┼──────┼──────┼──────┼───────┼──────┼────────┤
│ Prompt Eng.     │  ★   │  ★   │  ★   │  ★★   │  ★★  │   ★    │
│ Few-Shot        │  ★★  │  ★★  │  ★   │  ★★★  │  ★★★ │   ★★   │
│ RAG             │  ★★★ │  ★★★ │  ★★  │  ★★★★ │  ★★★ │  ★★★★  │
│ Fine-Tuning     │ ★★★★ │ ★★★★ │  ★★★ │ ★★★★★ │★★★★★ │ ★★★★★  │
│ Pre-training    │★★★★★│★★★★★ │★★★★★ │ ★★★★★ │★★★★★ │ ★★★★★  │
└─────────────────┴──────┴──────┴──────┴───────┴──────┴────────┘

Légende:
★ = Faible/Peu | ★★★★★ = Élevé/Beaucoup

Data   = Quantité de données requises
Cost   = Coût en compute/argent
Time   = Temps de développement
Perf   = Performance finale
Custom = Niveau de customisation possible
Domain = Adaptation au domaine spécifique
        """

        return table

    def decision_tree(self) -> str:
        """Arbre de décision pour choisir l'approche"""

        tree = """
=== Arbre de Décision: Quelle Approche Choisir? ===

Combien de données avez-vous?
│
├─ < 10 exemples
│  └─> Prompt Engineering ou Few-Shot Learning
│
├─ 10-100 exemples
│  │
│  ├─ Budget faible?
│  │  └─> Few-Shot Learning
│  │
│  └─ Budget moyen/élevé?
│     └─> RAG ou Fine-Tuning (si données haute qualité)
│
├─ 100-1000 exemples
│  │
│  ├─ Besoin performance excellente?
│  │  └─> Fine-Tuning
│  │
│  └─ Performance bonne suffit?
│     └─> RAG
│
└─ > 1000 exemples
   │
   ├─ Budget élevé + Expertise avancée?
   │  └─> Fine-Tuning (fortement recommandé)
   │
   ├─ Budget moyen?
   │  └─> Fine-Tuning ou RAG (selon cas d'usage)
   │
   └─ Budget faible?
      └─> RAG

Cas spéciaux:
• Informations changeantes fréquemment? → RAG
• Besoin de citations/sources? → RAG
• Domaine très spécifique (médical, légal)? → Fine-Tuning
• Ton/style très particulier? → Fine-Tuning
• Nouveau language/domaine inexploré? → Pre-training
        """

        return tree


# Exemple d'utilisation
if __name__ == "__main__":
    comparison = AdaptationComparison()

    print(comparison.generate_comparison_table())
    print("\n" + "="*60 + "\n")
    print(comparison.decision_tree())
    print("\n" + "="*60 + "\n")

    # Recommandations
    scenarios = [
        {
            "name": "Startup avec 50 exemples",
            "data_size": 50,
            "budget": "low",
            "performance_need": "good",
            "expertise": "beginner"
        },
        {
            "name": "Entreprise avec 5000 exemples",
            "data_size": 5000,
            "budget": "high",
            "performance_need": "excellent",
            "expertise": "advanced"
        },
        {
            "name": "Prototypage rapide",
            "data_size": 5,
            "budget": "low",
            "performance_need": "basic",
            "expertise": "beginner"
        }
    ]

    print("=== Recommandations par Scénario ===\n")
    for scenario in scenarios:
        recommendation = comparison.recommend(
            scenario["data_size"],
            scenario["budget"],
            scenario["performance_need"],
            scenario["expertise"]
        )
        print(f"Scénario: {scenario['name']}")
        print(f"  Données: {scenario['data_size']} exemples")
        print(f"  Budget: {scenario['budget']}")
        print(f"  Recommandation: {recommendation}\n")
```

### 1.2 Quand Fine-tuner?

```python
"""
Critères de décision pour le fine-tuning
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class FineTuningDecision:
    """Aide à décider si le fine-tuning est approprié"""

    @staticmethod
    def should_finetune(
        data_size: int,
        data_quality: str,
        domain_specific: bool,
        performance_critical: bool,
        budget_available: bool,
        ml_expertise: bool
    ) -> Dict[str, any]:
        """
        Détermine si le fine-tuning est approprié

        Args:
            data_size: Nombre d'exemples de qualité
            data_quality: "low", "medium", "high"
            domain_specific: Domaine très spécifique?
            performance_critical: Performance critique?
            budget_available: Budget compute disponible?
            ml_expertise: Équipe avec expertise ML?

        Returns:
            Recommandation avec justification
        """

        score = 0
        reasons_for = []
        reasons_against = []

        # Critères positifs
        if data_size >= 100:
            score += 2
            reasons_for.append(f"Données suffisantes ({data_size} exemples)")
        elif data_size >= 50:
            score += 1
            reasons_for.append(f"Données marginales ({data_size} exemples)")
        else:
            score -= 2
            reasons_against.append(f"Données insuffisantes ({data_size} < 100)")

        if data_quality == "high":
            score += 2
            reasons_for.append("Données de haute qualité")
        elif data_quality == "medium":
            score += 1
        else:
            score -= 1
            reasons_against.append("Données de faible qualité")

        if domain_specific:
            score += 2
            reasons_for.append("Domaine très spécifique nécessitant adaptation")

        if performance_critical:
            score += 2
            reasons_for.append("Performance critique pour le business")

        if budget_available:
            score += 1
            reasons_for.append("Budget compute disponible")
        else:
            score -= 1
            reasons_against.append("Budget compute limité")

        if ml_expertise:
            score += 1
            reasons_for.append("Expertise ML en interne")
        else:
            score -= 1
            reasons_against.append("Manque d'expertise ML")

        # Décision
        if score >= 5:
            recommendation = "Fine-tuning fortement recommandé"
            confidence = "high"
        elif score >= 3:
            recommendation = "Fine-tuning recommandé"
            confidence = "medium"
        elif score >= 1:
            recommendation = "Fine-tuning possible mais risqué"
            confidence = "low"
        else:
            recommendation = "Fine-tuning déconseillé - considérer alternatives"
            confidence = "high"

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "score": score,
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "alternatives": FineTuningDecision._suggest_alternatives(score)
        }

    @staticmethod
    def _suggest_alternatives(score: int) -> List[str]:
        """Suggère des alternatives si fine-tuning pas optimal"""

        if score < 1:
            return [
                "Prompt Engineering avec few-shot examples",
                "RAG avec vector database",
                "API externe spécialisée"
            ]
        elif score < 3:
            return [
                "Commencer par RAG",
                "Collecter plus de données puis fine-tuner",
                "Tester avec modèle plus petit d'abord"
            ]
        else:
            return []

    @staticmethod
    def common_mistakes() -> List[Dict[str, str]]:
        """Erreurs communes en fine-tuning"""

        return [
            {
                "mistake": "Fine-tuner avec trop peu de données",
                "consequence": "Overfitting sévère, modèle inutilisable",
                "solution": "Minimum 100 exemples, idéalement 1000+"
            },
            {
                "mistake": "Données de mauvaise qualité",
                "consequence": "Modèle apprend les erreurs",
                "solution": "Curation manuelle, validation rigoureuse"
            },
            {
                "mistake": "Fine-tuner pour une tâche que le modèle fait déjà",
                "consequence": "Gaspillage de ressources",
                "solution": "Tester prompt engineering d'abord"
            },
            {
                "mistake": "Pas de validation set",
                "consequence": "Pas de détection d'overfitting",
                "solution": "Toujours avoir train/val/test split"
            },
            {
                "mistake": "Learning rate trop élevé",
                "consequence": "Catastrophic forgetting",
                "solution": "LR faible (1e-5 à 5e-5) pour fine-tuning"
            },
            {
                "mistake": "Fine-tuner trop longtemps",
                "consequence": "Overfitting, perte des capacités générales",
                "solution": "Early stopping, monitoring validation loss"
            },
            {
                "mistake": "Ignorer class imbalance",
                "consequence": "Modèle biaisé vers classe majoritaire",
                "solution": "Balancer le dataset ou weighted loss"
            }
        ]


# Exemple d'utilisation
if __name__ == "__main__":
    decision = FineTuningDecision()

    # Scénario 1: Startup avec peu de données
    print("=== Scénario 1: Startup Healthcare ===")
    result1 = decision.should_finetune(
        data_size=50,
        data_quality="medium",
        domain_specific=True,
        performance_critical=True,
        budget_available=False,
        ml_expertise=False
    )

    print(f"Recommandation: {result1['recommendation']}")
    print(f"Confiance: {result1['confidence']}")
    print(f"Score: {result1['score']}")
    print(f"\nRaisons pour:")
    for reason in result1['reasons_for']:
        print(f"  ✅ {reason}")
    print(f"\nRaisons contre:")
    for reason in result1['reasons_against']:
        print(f"  ❌ {reason}")
    print(f"\nAlternatives suggérées:")
    for alt in result1['alternatives']:
        print(f"  → {alt}")

    print("\n" + "="*60 + "\n")

    # Scénario 2: Entreprise avec beaucoup de données
    print("=== Scénario 2: Grande Entreprise ===")
    result2 = decision.should_finetune(
        data_size=5000,
        data_quality="high",
        domain_specific=True,
        performance_critical=True,
        budget_available=True,
        ml_expertise=True
    )

    print(f"Recommandation: {result2['recommendation']}")
    print(f"Confiance: {result2['confidence']}")
    print(f"Score: {result2['score']}")

    print("\n" + "="*60 + "\n")

    # Erreurs communes
    print("=== Erreurs Communes en Fine-Tuning ===\n")
    for mistake in decision.common_mistakes():
        print(f"❌ {mistake['mistake']}")
        print(f"   Conséquence: {mistake['consequence']}")
        print(f"   Solution: {mistake['solution']}\n")
```

---

## Types de Fine-tuning

### 2.1 Full Fine-tuning vs Parameter-Efficient

```python
"""
Différents types de fine-tuning
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FineTuningType:
    """Caractéristiques d'un type de fine-tuning"""
    name: str
    description: str

    # Paramètres
    params_updated: str  # "all" ou "subset"
    memory_requirement: int  # 1-5
    compute_requirement: int  # 1-5

    # Performance
    effectiveness: int  # 1-5
    training_speed: int  # 1-5

    # Cas d'usage
    best_for: List[str]
    gpu_required: str


class FineTuningTypes:
    """Comparaison des types de fine-tuning"""

    @staticmethod
    def get_types() -> Dict[str, FineTuningType]:
        """Retourne tous les types de fine-tuning"""

        return {
            "full": FineTuningType(
                name="Full Fine-Tuning",
                description="Mise à jour de tous les paramètres du modèle",

                params_updated="all (100%)",
                memory_requirement=5,
                compute_requirement=5,

                effectiveness=5,
                training_speed=1,

                best_for=[
                    "Adaptation forte au domaine",
                    "Données abondantes (>10k exemples)",
                    "Budget compute élevé",
                    "Performance maximale requise"
                ],
                gpu_required="Multiple A100 ou H100"
            ),

            "lora": FineTuningType(
                name="LoRA (Low-Rank Adaptation)",
                description="Ajoute des matrices low-rank trainables",

                params_updated="~0.1-1% (matrices LoRA uniquement)",
                memory_requirement=2,
                compute_requirement=2,

                effectiveness=4,
                training_speed=4,

                best_for=[
                    "Budget compute limité",
                    "Données moyennes (100-10k)",
                    "GPU consumer (RTX 3090, 4090)",
                    "Multiple tasks (multiple adapters)"
                ],
                gpu_required="1x RTX 3090/4090 ou A10"
            ),

            "qlora": FineTuningType(
                name="QLoRA (Quantized LoRA)",
                description="LoRA + quantization 4-bit du base model",

                params_updated="~0.1-1% (matrices LoRA)",
                memory_requirement=1,
                compute_requirement=1,

                effectiveness=4,
                training_speed=3,

                best_for=[
                    "GPU très limité",
                    "Fine-tuner gros modèles (70B) sur 1 GPU",
                    "Prototypage rapide",
                    "Budget très serré"
                ],
                gpu_required="1x RTX 3090 (peut fine-tuner 70B!)"
            ),

            "prefix_tuning": FineTuningType(
                name="Prefix Tuning",
                description="Ajoute des tokens virtuels apprenables",

                params_updated="<0.1% (prefixes)",
                memory_requirement=2,
                compute_requirement=2,

                effectiveness=3,
                training_speed=4,

                best_for=[
                    "Tâches spécifiques",
                    "Peu de données",
                    "Multiple tasks"
                ],
                gpu_required="1x RTX 3090"
            ),

            "adapter": FineTuningType(
                name="Adapter Layers",
                description="Insère des couches adapter trainables",

                params_updated="~2-5% (adapters)",
                memory_requirement=3,
                compute_requirement=3,

                effectiveness=4,
                training_speed=3,

                best_for=[
                    "Multiple domaines/langues",
                    "Modular adaptation",
                    "Transfer learning"
                ],
                gpu_required="1-2x A100"
            )
        }

    @staticmethod
    def comparison_matrix() -> str:
        """Matrice de comparaison"""

        return """
=== Comparaison des Types de Fine-Tuning ===

┌─────────────────┬────────┬────────┬────────┬──────┬────────┐
│ Type            │ Params │ Memory │ Compute│ Perf │ Speed  │
├─────────────────┼────────┼────────┼────────┼──────┼────────┤
│ Full FT         │  100%  │ ★★★★★  │ ★★★★★  │★★★★★ │   ★    │
│ LoRA            │   ~1%  │   ★★   │   ★★   │ ★★★★ │  ★★★★  │
│ QLoRA           │   ~1%  │   ★    │   ★    │ ★★★★ │  ★★★   │
│ Prefix Tuning   │  <0.1% │   ★★   │   ★★   │  ★★★ │  ★★★★  │
│ Adapter Layers  │   ~3%  │  ★★★   │  ★★★   │ ★★★★ │  ★★★   │
└─────────────────┴────────┴────────┴────────┴──────┴────────┘

Recommandations:
• Budget illimité + Performance max → Full Fine-Tuning
• Budget moyen + Bon équilibre → LoRA
• GPU limité + Gros modèles → QLoRA
• Multiple tasks → LoRA ou Adapters
• Prototypage rapide → QLoRA

Note: LoRA Chapter 8 couvre LoRA/QLoRA en détail
Ce chapitre se concentre sur Full Fine-Tuning
        """


# Exemple d'utilisation
if __name__ == "__main__":
    types = FineTuningTypes()

    print(types.comparison_matrix())

    print("\n" + "="*60 + "\n")

    # Détails de chaque type
    for type_key, ft_type in types.get_types().items():
        print(f"=== {ft_type.name} ===")
        print(f"{ft_type.description}\n")
        print(f"Paramètres mis à jour: {ft_type.params_updated}")
        print(f"GPU requis: {ft_type.gpu_required}")
        print(f"\nMeilleur pour:")
        for use_case in ft_type.best_for:
            print(f"  • {use_case}")
        print()
```

*[Suite du chapitre dans le prochain message...]*

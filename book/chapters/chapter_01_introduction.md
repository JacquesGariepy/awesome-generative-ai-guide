# Chapitre 1: Introduction à l'IA Générative et aux LLMs

## 🚀 Bienvenue dans l'Ère de l'IA Générative

> *"The hottest new programming language is English."* — Andrej Karpathy, ex-Director of AI chez Tesla

Nous sommes à l'aube d'une révolution technologique comparable à l'invention d'Internet. L'**Intelligence Artificielle Générative** et les **Large Language Models (LLMs)** ne sont pas juste une autre tendance tech - ils transforment fondamentalement comment nous créons, travaillons, et interagissons avec les machines.

### Le Moment ChatGPT

**30 novembre 2022**: OpenAI lance ChatGPT. En 5 jours, 1 million d'utilisateurs. En 2 mois, 100 millions d'utilisateurs - la croissance la plus rapide de l'histoire technologique.

Pourquoi ce succès explosif? Pour la première fois, **n'importe qui** peut avoir une conversation naturelle avec une IA. Plus besoin de coder, de commandes spéciales, ou de formation technique. Vous parlez, l'IA répond - intelligemment.

```python
"""
Un exemple qui a changé le monde
"""

# Avant ChatGPT (2021)
def translate_text(text, target_lang):
    # Nécessite API, credentials, code...
    from googletrans import Translator
    translator = Translator()
    return translator.translate(text, dest=target_lang).text

# Après ChatGPT (2022+)
prompt = "Translate to French: Hello, how are you?"
response = chatgpt(prompt)
# → "Bonjour, comment allez-vous ?"

# Pas de setup, pas de code complexe, pas de formation
# Juste... parler naturellement
```

## 1.1 Qu'est-ce que l'IA Générative?

### Définition

L'**IA Générative** est une famille de modèles d'IA capables de **créer du contenu nouveau** (texte, images, code, audio, vidéo) plutôt que simplement classifier ou prédire.

**Distinction clé**:

| IA Classique (Discriminative) | IA Générative |
|-------------------------------|---------------|
| **Analyse** des données existantes | **Crée** de nouvelles données |
| Classification, prédiction | Génération, création |
| "Ceci est-il un chat?" | "Génère une image de chat" |
| **Exemple**: Spam detector | **Exemple**: ChatGPT, DALL-E |

### Les 3 Piliers de l'IA Générative

```python
"""
Les 3 types d'IA Générative en 2024-2026
"""

from enum import Enum
from typing import List, Dict


class GenerativeAIType(Enum):
    """Types d'IA Générative"""
    TEXT = "text"  # LLMs: GPT-4, Claude, Llama
    IMAGE = "image"  # DALL-E, Midjourney, Stable Diffusion
    MULTIMODAL = "multimodal"  # GPT-4V, Gemini, Claude 3


class AIGenerationExample:
    """Exemples concrets par type"""

    @staticmethod
    def text_examples() -> List[str]:
        """Applications LLM"""
        return [
            "Chatbots et assistants (ChatGPT, Claude)",
            "Génération de code (GitHub Copilot, Cursor)",
            "Rédaction et marketing (Jasper, Copy.ai)",
            "Résumé et analyse de documents",
            "Traduction contextuelle",
            "Customer support automatisé",
            "Génération de scripts, emails, rapports"
        ]

    @staticmethod
    def image_examples() -> List[str]:
        """Applications image"""
        return [
            "Art génératif (Midjourney)",
            "Design et prototypage (Canva AI)",
            "Photo editing (Photoshop AI)",
            "Game assets generation",
            "Marketing visuals",
            "Product mockups"
        ]

    @staticmethod
    def multimodal_examples() -> List[str]:
        """Applications multimodales"""
        return [
            "Visual Question Answering",
            "Document understanding (PDFs, images)",
            "Video analysis",
            "Accessibility (image → description)",
            "Medical imaging analysis",
            "Autonomous vehicles"
        ]


# Impact économique
MARKET_SIZE = {
    "2023": "43.87 billion USD",
    "2030": "1.3 trillion USD (projected)",  # CAGR 42%
    "growth_rate": "42% CAGR"
}

# Les plus grandes companies investissent massivement
INVESTMENTS_2023 = {
    "Microsoft": "10+ billion (OpenAI)",
    "Google": "70+ billion (AI R&D)",
    "Meta": "27 billion (AI infrastructure)",
    "Amazon": "75+ billion (AWS + AI)",
    "Anthropic": "5+ billion raised"
}


if __name__ == "__main__":
    print("="*60)
    print("L'ÈRE DE L'IA GÉNÉRATIVE")
    print("="*60)
    print()

    examples = AIGenerationExample()

    print("### Applications LLM (Texte)")
    for app in examples.text_examples()[:5]:
        print(f"  ✓ {app}")

    print("\n### Applications Image")
    for app in examples.image_examples()[:4]:
        print(f"  ✓ {app}")

    print("\n### Applications Multimodales")
    for app in examples.multimodal_examples()[:4]:
        print(f"  ✓ {app}")

    print("\n" + "="*60)
    print("IMPACT ÉCONOMIQUE")
    print("="*60)
    print(f"\nMarché 2023: {MARKET_SIZE['2023']}")
    print(f"Projection 2030: {MARKET_SIZE['2030']}")
    print(f"Croissance: {MARKET_SIZE['growth_rate']}")
```

## 1.2 Histoire et Évolution des LLMs

### Timeline: De Turing à ChatGPT

```python
"""
Timeline de l'évolution des LLMs
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class LLMHistory:
    """Événement historique dans l'évolution des LLMs"""
    year: int
    name: str
    organization: str
    parameters: Optional[str]
    breakthrough: str
    impact: int  # 1-10


# Timeline complète
LLM_TIMELINE = [
    LLMHistory(
        year=1950,
        name="Turing Test",
        organization="Alan Turing",
        parameters=None,
        breakthrough="Concept: machines peuvent-elles penser?",
        impact=10
    ),

    LLMHistory(
        year=1957,
        name="Perceptron",
        organization="Frank Rosenblatt",
        parameters="1 layer",
        breakthrough="Premier réseau de neurones artificiel",
        impact=8
    ),

    LLMHistory(
        year=1997,
        name="LSTM",
        organization="Hochreiter & Schmidhuber",
        parameters=None,
        breakthrough="Mémoire long terme pour séquences",
        impact=9
    ),

    LLMHistory(
        year=2013,
        name="Word2Vec",
        organization="Google (Mikolov et al.)",
        parameters="~100M",
        breakthrough="Word embeddings - mots comme vecteurs",
        impact=9
    ),

    LLMHistory(
        year=2017,
        name="Transformer",
        organization="Google Brain (Vaswani et al.)",
        parameters="65M",
        breakthrough="'Attention Is All You Need' - révolution",
        impact=10
    ),

    LLMHistory(
        year=2018,
        name="BERT",
        organization="Google AI",
        parameters="340M",
        breakthrough="Bidirectional understanding",
        impact=9
    ),

    LLMHistory(
        year=2018,
        name="GPT-1",
        organization="OpenAI",
        parameters="117M",
        breakthrough="Generative pre-training",
        impact=7
    ),

    LLMHistory(
        year=2019,
        name="GPT-2",
        organization="OpenAI",
        parameters="1.5B",
        breakthrough="'Too dangerous to release' (initially)",
        impact=8
    ),

    LLMHistory(
        year=2020,
        name="GPT-3",
        organization="OpenAI",
        parameters="175B",
        breakthrough="Few-shot learning, emergent abilities",
        impact=10
    ),

    LLMHistory(
        year=2021,
        name="Codex",
        organization="OpenAI",
        parameters="12B",
        breakthrough="Code generation (GitHub Copilot)",
        impact=9
    ),

    LLMHistory(
        year=2022,
        name="ChatGPT",
        organization="OpenAI",
        parameters="175B (GPT-3.5)",
        breakthrough="RLHF, conversational AI mainstream",
        impact=10
    ),

    LLMHistory(
        year=2023,
        name="GPT-4",
        organization="OpenAI",
        parameters="1.7T (rumored)",
        breakthrough="Multimodal, reasoning, tool use",
        impact=10
    ),

    LLMHistory(
        year=2023,
        name="Llama 2",
        organization="Meta",
        parameters="7B, 13B, 70B",
        breakthrough="Open-source SOTA, commercial use",
        impact=10
    ),

    LLMHistory(
        year=2023,
        name="Claude 2",
        organization="Anthropic",
        parameters="Unknown",
        breakthrough="100k context, Constitutional AI",
        impact=9
    ),

    LLMHistory(
        year=2024,
        name="GPT-4 Turbo",
        organization="OpenAI",
        parameters="Unknown",
        breakthrough="128k context, cheaper, faster",
        impact=9
    ),

    LLMHistory(
        year=2024,
        name="Claude 3 (Opus, Sonnet, Haiku)",
        organization="Anthropic",
        parameters="Unknown",
        breakthrough="Multimodal, beats GPT-4 on many benchmarks",
        impact=10
    ),

    LLMHistory(
        year=2024,
        name="Gemini Ultra",
        organization="Google DeepMind",
        parameters="Unknown",
        breakthrough="Native multimodal, integrates with Google",
        impact=9
    ),

    LLMHistory(
        year=2024,
        name="Mixtral 8x7B",
        organization="Mistral AI",
        parameters="46.7B (sparse)",
        breakthrough="MoE, open-source, beats GPT-3.5",
        impact=9
    )
]


def print_timeline():
    """Print formatted timeline"""
    print("="*80)
    print("TIMELINE: L'ÉVOLUTION DES LLMs")
    print("="*80)
    print()

    # Group by decade
    current_decade = None

    for event in LLM_TIMELINE:
        decade = (event.year // 10) * 10

        if decade != current_decade:
            print(f"\n### {decade}s")
            print("-" * 80)
            current_decade = decade

        # Impact visualization
        impact_stars = "⭐" * event.impact

        print(f"\n**{event.year}**: {event.name}")
        print(f"  Organization: {event.organization}")
        if event.parameters:
            print(f"  Parameters: {event.parameters}")
        print(f"  Breakthrough: {event.breakthrough}")
        print(f"  Impact: {impact_stars}")


if __name__ == "__main__":
    print_timeline()

    print("\n" + "="*80)
    print("OBSERVATIONS CLÉS")
    print("="*80)
    print("""
    1. **Accélération exponentielle**: 50 ans (1950-2000) → 5 ans (2018-2023)

    2. **Scale is all you need?**: De 117M (GPT-1) à 1.7T (GPT-4) en 5 ans

    3. **Open-source vs Closed**: Llama 2, Mixtral démocratisent l'accès

    4. **Émergence de capacités**: À partir de ~10B params, emergent abilities

    5. **Multimodal is the future**: GPT-4V, Claude 3, Gemini

    6. **Context windows**: 2k (GPT-3) → 100k (Claude 2) → 200k (Claude 3)

    7. **RLHF révolution**: ChatGPT montre l'importance de l'alignment
    """)
```

## 1.3 Principaux Acteurs et Leurs Modèles

### Le Paysage LLM en 2024-2026

```python
"""
Comparaison des principaux acteurs et modèles LLM
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ModelType(Enum):
    """Type de modèle"""
    CLOSED_SOURCE = "closed_source"  # API only
    OPEN_WEIGHTS = "open_weights"  # Weights téléchargeables
    OPEN_SOURCE = "open_source"  # Weights + training code


@dataclass
class LLMModel:
    """Caractéristiques d'un modèle LLM"""
    name: str
    organization: str
    release_date: str
    parameters: str
    context_window: int  # tokens
    model_type: ModelType
    strengths: List[str]
    pricing: Optional[str]  # Pour les APIs
    benchmark_score: float  # MMLU ou average


# Les Big Players
LLM_LANDSCAPE_2024 = {
    "OpenAI": [
        LLMModel(
            name="GPT-4 Turbo",
            organization="OpenAI",
            release_date="Nov 2023",
            parameters="Unknown (rumored 1.7T)",
            context_window=128000,
            model_type=ModelType.CLOSED_SOURCE,
            strengths=[
                "Reasoning et problem-solving",
                "Code generation (best-in-class)",
                "Creative writing",
                "Multimodal (vision)",
                "Function calling"
            ],
            pricing="$0.01/1K input tokens, $0.03/1K output",
            benchmark_score=86.4  # MMLU
        ),
        LLMModel(
            name="GPT-3.5 Turbo",
            organization="OpenAI",
            release_date="Nov 2023",
            parameters="Unknown",
            context_window=16385,
            model_type=ModelType.CLOSED_SOURCE,
            strengths=[
                "Rapide et pas cher",
                "Conversational",
                "Good for most tasks"
            ],
            pricing="$0.0005/1K input, $0.0015/1K output",
            benchmark_score=70.0
        )
    ],

    "Anthropic": [
        LLMModel(
            name="Claude 3 Opus",
            organization="Anthropic",
            release_date="Mar 2024",
            parameters="Unknown",
            context_window=200000,
            model_type=ModelType.CLOSED_SOURCE,
            strengths=[
                "Longest context (200k)",
                "Best at analysis and reasoning",
                "Multimodal (vision)",
                "Beats GPT-4 on many benchmarks",
                "Constitutional AI (safer)"
            ],
            pricing="$15/1M input, $75/1M output",
            benchmark_score=86.8  # MMLU (beats GPT-4!)
        ),
        LLMModel(
            name="Claude 3 Sonnet",
            organization="Anthropic",
            release_date="Mar 2024",
            parameters="Unknown",
            context_window=200000,
            model_type=ModelType.CLOSED_SOURCE,
            strengths=[
                "Balance performance/cost",
                "Still beats GPT-3.5",
                "200k context"
            ],
            pricing="$3/1M input, $15/1M output",
            benchmark_score=79.0
        )
    ],

    "Google": [
        LLMModel(
            name="Gemini Ultra",
            organization="Google DeepMind",
            release_date="Dec 2023",
            parameters="Unknown",
            context_window=32000,
            model_type=ModelType.CLOSED_SOURCE,
            strengths=[
                "Native multimodal (not bolted-on)",
                "Integrates with Google ecosystem",
                "Strong at math and science"
            ],
            pricing="Not publicly available yet",
            benchmark_score=90.0  # MMLU (claimed)
        )
    ],

    "Meta": [
        LLMModel(
            name="Llama 2 70B",
            organization="Meta",
            release_date="Jul 2023",
            parameters="70B",
            context_window=4096,
            model_type=ModelType.OPEN_WEIGHTS,
            strengths=[
                "Open-source (commercial use OK)",
                "Can run locally",
                "Fine-tunable",
                "Community support",
                "Competitive with GPT-3.5"
            ],
            pricing="Free (self-host)",
            benchmark_score=68.9
        ),
        LLMModel(
            name="Llama 2 7B/13B",
            organization="Meta",
            release_date="Jul 2023",
            parameters="7B / 13B",
            context_window=4096,
            model_type=ModelType.OPEN_WEIGHTS,
            strengths=[
                "Runs on consumer hardware",
                "Great for fine-tuning",
                "Fast inference"
            ],
            pricing="Free",
            benchmark_score=45.9  # 7B
        )
    ],

    "Mistral AI": [
        LLMModel(
            name="Mixtral 8x7B",
            organization="Mistral AI",
            release_date="Dec 2023",
            parameters="46.7B (8 experts x 7B, sparse)",
            context_window=32000,
            model_type=ModelType.OPEN_WEIGHTS,
            strengths=[
                "Mixture of Experts (MoE)",
                "Beats GPT-3.5 on most benchmarks",
                "Open-source",
                "32k context",
                "Apache 2.0 license"
            ],
            pricing="Free (self-host) or €2/1M tokens (API)",
            benchmark_score=70.6
        )
    ]
}


def print_llm_landscape():
    """Print formatted LLM landscape"""
    print("="*80)
    print("PAYSAGE LLM 2024: LES ACTEURS MAJEURS")
    print("="*80)
    print()

    for company, models in LLM_LANDSCAPE_2024.items():
        print(f"\n### {company}")
        print("-" * 80)

        for model in models:
            print(f"\n**{model.name}**")
            print(f"  Release: {model.release_date}")
            print(f"  Parameters: {model.parameters}")
            print(f"  Context: {model.context_window:,} tokens")
            print(f"  Type: {model.model_type.value}")
            print(f"  MMLU Score: {model.benchmark_score}/100")
            if model.pricing:
                print(f"  Pricing: {model.pricing}")

            print("  Strengths:")
            for strength in model.strengths[:3]:
                print(f"    ✓ {strength}")


if __name__ == "__main__":
    print_llm_landscape()

    print("\n" + "="*80)
    print("COMMENT CHOISIR?")
    print("="*80)
    print("""
    **Pour la performance maximale**:
    → GPT-4 Turbo ou Claude 3 Opus

    **Pour le meilleur rapport qualité/prix**:
    → Claude 3 Sonnet ou GPT-3.5 Turbo

    **Pour l'open-source / self-hosting**:
    → Llama 2 70B ou Mixtral 8x7B

    **Pour les petits budgets / edge devices**:
    → Llama 2 7B ou 13B

    **Pour le contexte ultra-long (100k+ tokens)**:
    → Claude 3 (200k context)

    **Pour le code**:
    → GPT-4 Turbo (meilleur pour code)

    **Pour la safety / ethics**:
    → Claude 3 (Constitutional AI)
    """)
```

*[Suite avec applications, limitations, et premier projet dans la partie 2...]*

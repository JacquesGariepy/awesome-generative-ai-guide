# Chapitre 7 (Suite): Préparation des Données et Hyperparamètres

## Préparation des Données

La qualité de vos données détermine 80% du succès de votre fine-tuning. Cette section couvre tout ce qu'il faut savoir.

### 3.1 Format des Données

```python
"""
Formats de données pour fine-tuning
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class TrainingExample:
    """Exemple d'entraînement"""
    instruction: Optional[str]  # Instruction/prompt
    input: Optional[str]  # Input additionnel (optionnel)
    output: str  # Réponse attendue
    metadata: Optional[Dict] = None


class DatasetFormats:
    """Différents formats de datasets pour fine-tuning"""

    @staticmethod
    def completion_format() -> Dict[str, any]:
        """Format completion (simple)"""

        return {
            "name": "Completion Format",
            "description": "Format le plus simple: prompt + completion",
            "use_case": "Génération de texte, continuation",

            "example": {
                "prompt": "Translate to French: Hello, how are you?",
                "completion": "Bonjour, comment allez-vous?"
            },

            "jsonl_example": """
{"prompt": "Q: What is the capital of France?\\n\\nA:", "completion": " Paris"}
{"prompt": "Q: Who wrote Romeo and Juliet?\\n\\nA:", "completion": " William Shakespeare"}
            """,

            "pros": [
                "Format le plus simple",
                "Facile à créer",
                "Compatible avec OpenAI fine-tuning API"
            ],

            "cons": [
                "Peu structuré",
                "Difficile de séparer instruction/input/output",
                "Moins flexible"
            ]
        }

    @staticmethod
    def instruction_format() -> Dict[str, any]:
        """Format instruction (recommandé)"""

        return {
            "name": "Instruction Format",
            "description": "Format structuré avec instruction, input (optionnel), et output",
            "use_case": "Tâches variées, instruction following",

            "example": {
                "instruction": "Translate the following English text to French",
                "input": "Hello, how are you?",
                "output": "Bonjour, comment allez-vous?"
            },

            "jsonl_example": """
{"instruction": "Summarize the article", "input": "[Article text...]", "output": "[Summary]"}
{"instruction": "Classify sentiment", "input": "This movie was great!", "output": "positive"}
{"instruction": "Extract entities", "input": "Apple Inc. is in Cupertino", "output": "ORG: Apple Inc., LOC: Cupertino"}
            """,

            "pros": [
                "Structuré et clair",
                "Flexible (input optionnel)",
                "Facilite instruction following",
                "Standard dans la communauté"
            ],

            "cons": [
                "Légèrement plus complexe que completion",
                "Nécessite formatage"
            ]
        }

    @staticmethod
    def chat_format() -> Dict[str, any]:
        """Format chat/conversation"""

        return {
            "name": "Chat Format",
            "description": "Format conversationnel avec rôles (system, user, assistant)",
            "use_case": "Chatbots, assistants conversationnels",

            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather like?"},
                    {"role": "assistant", "content": "I don't have access to real-time weather data. Please check a weather service for current conditions."}
                ]
            },

            "jsonl_example": """
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "Why did the chicken cross the road..."}]}
            """,

            "pros": [
                "Idéal pour chatbots",
                "Support multi-turn conversations",
                "Séparation claire des rôles",
                "Compatible ChatGPT fine-tuning"
            ],

            "cons": [
                "Plus complexe",
                "Overhead de tokens (rôles répétés)",
                "Nécessite preprocessing spécifique"
            ]
        }

    @staticmethod
    def format_for_training(
        examples: List[TrainingExample],
        format_type: str = "instruction"
    ) -> List[str]:
        """
        Formate des exemples pour l'entraînement

        Args:
            examples: Liste de TrainingExample
            format_type: "completion", "instruction", ou "chat"

        Returns:
            Liste de strings formatés
        """

        formatted = []

        for ex in examples:
            if format_type == "completion":
                # Format simple prompt + completion
                text = f"{ex.instruction}"
                if ex.input:
                    text += f"\n{ex.input}"
                text += f"\n{ex.output}"
                formatted.append(text)

            elif format_type == "instruction":
                # Format Alpaca-style
                text = f"### Instruction:\n{ex.instruction}\n\n"
                if ex.input:
                    text += f"### Input:\n{ex.input}\n\n"
                text += f"### Response:\n{ex.output}"
                formatted.append(text)

            elif format_type == "chat":
                # Format chat
                messages = []
                if ex.instruction:
                    messages.append({"role": "user", "content": ex.instruction})
                if ex.input:
                    # Combiner instruction et input
                    user_msg = f"{ex.instruction}\n\n{ex.input}"
                    messages = [{"role": "user", "content": user_msg}]

                messages.append({"role": "assistant", "content": ex.output})
                formatted.append(json.dumps({"messages": messages}))

        return formatted


# Exemple d'utilisation
if __name__ == "__main__":
    formats_info = DatasetFormats()

    # Afficher les formats
    for format_func in [formats_info.completion_format,
                        formats_info.instruction_format,
                        formats_info.chat_format]:

        format_info = format_func()
        print(f"=== {format_info['name']} ===\n")
        print(f"{format_info['description']}")
        print(f"\nCas d'usage: {format_info['use_case']}\n")
        print(f"Exemple:")
        print(json.dumps(format_info['example'], indent=2))
        print(f"\nFormat JSONL:")
        print(format_info['jsonl_example'])
        print(f"\nAvantages:")
        for pro in format_info['pros']:
            print(f"  ✅ {pro}")
        print(f"\nInconvénients:")
        for con in format_info['cons']:
            print(f"  ❌ {con}")
        print("\n" + "="*60 + "\n")

    # Exemple de formatage
    examples = [
        TrainingExample(
            instruction="Translate to French",
            input="Hello world",
            output="Bonjour le monde"
        ),
        TrainingExample(
            instruction="Summarize this text",
            input="Long article about AI...",
            output="AI is transforming industries."
        )
    ]

    print("=== Exemples Formatés ===\n")

    for format_type in ["completion", "instruction", "chat"]:
        print(f"{format_type.upper()} FORMAT:")
        formatted = DatasetFormats.format_for_training(examples, format_type)
        for i, text in enumerate(formatted[:1], 1):  # Just show first example
            print(f"{text}\n")
        print("-" * 60 + "\n")
```

### 3.2 Data Quality Guidelines

```python
"""
Guidelines pour la qualité des données de fine-tuning
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import re


@dataclass
class QualityCheck:
    """Vérification de qualité"""
    name: str
    description: str
    check_function: callable
    severity: str  # "error", "warning", "info"


class DataQualityChecker:
    """Vérificateur de qualité de données"""

    def __init__(self):
        self.checks = self._initialize_checks()
        self.stats = {
            "total_examples": 0,
            "errors": 0,
            "warnings": 0,
            "passed": 0
        }

    def _initialize_checks(self) -> List[QualityCheck]:
        """Initialise les vérifications de qualité"""

        return [
            # Longueur minimale
            QualityCheck(
                name="minimum_length",
                description="Output doit avoir au moins 10 caractères",
                check_function=lambda ex: len(ex.get('output', '')) >= 10,
                severity="error"
            ),

            # Longueur maximale
            QualityCheck(
                name="maximum_length",
                description="Output ne doit pas dépasser 2048 tokens (~8000 chars)",
                check_function=lambda ex: len(ex.get('output', '')) <= 8000,
                severity="warning"
            ),

            # Pas de texte vide
            QualityCheck(
                name="non_empty_output",
                description="Output ne peut pas être vide",
                check_function=lambda ex: bool(ex.get('output', '').strip()),
                severity="error"
            ),

            # Cohérence instruction-output
            QualityCheck(
                name="instruction_present",
                description="Instruction doit être présente",
                check_function=lambda ex: bool(ex.get('instruction', '').strip()),
                severity="error"
            ),

            # Pas de repetitions excessives
            QualityCheck(
                name="no_excessive_repetition",
                description="Pas de répétitions excessives de mots",
                check_function=lambda ex: not DataQualityChecker._has_excessive_repetition(
                    ex.get('output', '')
                ),
                severity="warning"
            ),

            # Diversité du vocabulaire
            QualityCheck(
                name="vocabulary_diversity",
                description="Ratio unique words / total words > 0.3",
                check_function=lambda ex: DataQualityChecker._vocab_diversity(
                    ex.get('output', '')
                ) > 0.3,
                severity="info"
            ),

            # Pas de contenu offensant (placeholder)
            QualityCheck(
                name="no_offensive_content",
                description="Pas de langage offensant",
                check_function=lambda ex: not DataQualityChecker._contains_offensive(
                    ex.get('output', '')
                ),
                severity="error"
            ),

            # Format correct (si JSON)
            QualityCheck(
                name="valid_json_if_applicable",
                description="Si output est JSON, doit être valide",
                check_function=lambda ex: DataQualityChecker._check_json_if_applicable(
                    ex.get('output', '')
                ),
                severity="warning"
            )
        ]

    @staticmethod
    def _has_excessive_repetition(text: str) -> bool:
        """Détecte les répétitions excessives"""

        words = text.lower().split()
        if len(words) < 5:
            return False

        # Vérifier triplets consécutifs identiques
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True

        return False

    @staticmethod
    def _vocab_diversity(text: str) -> float:
        """Calcule la diversité du vocabulaire"""

        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    @staticmethod
    def _contains_offensive(text: str) -> bool:
        """Vérifie le contenu offensant (placeholder)"""

        # En production, utiliser une vraie liste de termes offensants
        # ou un modèle de modération comme OpenAI Moderation API

        offensive_patterns = [
            # Liste simplifiée pour l'exemple
        ]

        text_lower = text.lower()
        return any(pattern in text_lower for pattern in offensive_patterns)

    @staticmethod
    def _check_json_if_applicable(text: str) -> bool:
        """Vérifie la validité JSON si applicable"""

        import json

        # Si le texte semble être du JSON
        if text.strip().startswith(('{', '[')):
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                return False

        return True  # Pas JSON, donc OK

    def check_example(self, example: Dict) -> Dict[str, any]:
        """Vérifie un exemple"""

        result = {
            "example": example,
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": []
        }

        for check in self.checks:
            try:
                if not check.check_function(example):
                    if check.severity == "error":
                        result["errors"].append(check.description)
                        result["passed"] = False
                    elif check.severity == "warning":
                        result["warnings"].append(check.description)
                    else:  # info
                        result["info"].append(check.description)
            except Exception as e:
                result["warnings"].append(f"Check {check.name} failed: {str(e)}")

        return result

    def check_dataset(self, dataset: List[Dict]) -> Dict[str, any]:
        """Vérifie un dataset complet"""

        results = []
        error_count = 0
        warning_count = 0

        for i, example in enumerate(dataset):
            result = self.check_example(example)
            results.append(result)

            if result["errors"]:
                error_count += 1
            if result["warnings"]:
                warning_count += 1

        self.stats["total_examples"] = len(dataset)
        self.stats["errors"] = error_count
        self.stats["warnings"] = warning_count
        self.stats["passed"] = len(dataset) - error_count

        return {
            "stats": self.stats,
            "results": results,
            "pass_rate": (len(dataset) - error_count) / len(dataset) if dataset else 0
        }

    def generate_report(self, check_results: Dict) -> str:
        """Génère un rapport de qualité"""

        stats = check_results["stats"]

        report = ["=== Rapport de Qualité des Données ===\n"]
        report.append(f"Total d'exemples: {stats['total_examples']}")
        report.append(f"Exemples valides: {stats['passed']} ({stats['passed']/stats['total_examples']:.1%})")
        report.append(f"Exemples avec erreurs: {stats['errors']} ({stats['errors']/stats['total_examples']:.1%})")
        report.append(f"Exemples avec warnings: {stats['warnings']} ({stats['warnings']/stats['total_examples']:.1%})")

        # Trouver les erreurs les plus fréquentes
        error_types = {}
        warning_types = {}

        for result in check_results["results"]:
            for error in result["errors"]:
                error_types[error] = error_types.get(error, 0) + 1
            for warning in result["warnings"]:
                warning_types[warning] = warning_types.get(warning, 0) + 1

        if error_types:
            report.append("\n### Erreurs Fréquentes:")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  ❌ {error}: {count} occurrences")

        if warning_types:
            report.append("\n### Warnings Fréquents:")
            for warning, count in sorted(warning_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  ⚠️  {warning}: {count} occurrences")

        return "\n".join(report)


# Exemple d'utilisation
if __name__ == "__main__":
    checker = DataQualityChecker()

    # Dataset d'exemple
    dataset = [
        {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        },
        {
            "instruction": "Summarize",
            "input": "Long text...",
            "output": ""  # Erreur: output vide
        },
        {
            "instruction": "Generate JSON",
            "input": "user data",
            "output": '{"name": "invalid json'  # Erreur: JSON invalide
        },
        {
            "instruction": "Write a poem",
            "input": "",
            "output": "Roses are red, red, red, red, red"  # Warning: répétition
        },
        {
            "instruction": "Explain AI",
            "input": "",
            "output": "AI AI AI AI AI AI AI"  # Multiple warnings
        }
    ]

    # Vérifier le dataset
    check_results = checker.check_dataset(dataset)

    # Générer le rapport
    print(checker.generate_report(check_results))

    print("\n" + "="*60 + "\n")

    # Détails des exemples problématiques
    print("=== Exemples Problématiques ===\n")
    for i, result in enumerate(check_results["results"]):
        if result["errors"] or result["warnings"]:
            print(f"Exemple {i+1}:")
            if result["errors"]:
                print(f"  Erreurs: {result['errors']}")
            if result["warnings"]:
                print(f"  Warnings: {result['warnings']}")
            print()
```

*[Suite avec hyperparamètres dans le prochain message...]*

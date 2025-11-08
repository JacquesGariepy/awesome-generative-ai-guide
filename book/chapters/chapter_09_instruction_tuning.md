# Chapitre 9: Instruction Tuning

## Introduction

L'**Instruction Tuning** est la technique qui transforme un modèle de langage brut (pré-entraîné) en un assistant capable de suivre des instructions humaines.

### Pourquoi l'Instruction Tuning?

```python
"""
Comparaison: Base Model vs Instruction-Tuned Model
"""

# ❌ Base Model (pre-trained only)
prompt = "Translate to French: Hello, how are you?"
base_output = "Translate to Spanish: Hola, ¿cómo estás? Translate to German: Hallo, wie geht es dir?"
# → Continue le pattern, ne suit PAS l'instruction

# ✅ Instruction-Tuned Model
prompt = "Translate to French: Hello, how are you?"
tuned_output = "Bonjour, comment allez-vous ?"
# → Suit l'instruction!
```

### Différence avec Fine-Tuning Classique

| Aspect | Task-Specific Fine-Tuning | Instruction Tuning |
|--------|---------------------------|-------------------|
| **Objectif** | 1 tâche spécifique | Suivre des instructions variées |
| **Dataset** | 1 format (e.g., Q&A) | Multi-tasks (traduction, résumé, etc.) |
| **Output** | Modèle spécialisé | Assistant généraliste |
| **Exemples** | 10k+ pour 1 tâche | 10k+ pour N tâches |
| **Transfert** | Limité | Excellent (zero-shot) |

## 1. Formats de Données pour Instruction Tuning

```python
"""
Formats standards pour instruction tuning

Les plus populaires:
1. Alpaca Format (Stanford)
2. ShareGPT Format (multi-turn)
3. OpenAI Format (chat completion)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import json


class InstructionFormat(Enum):
    """Types de formats d'instruction"""
    ALPACA = "alpaca"  # Stanford Alpaca
    SHAREGPT = "sharegpt"  # Conversations multi-turn
    OPENAI = "openai"  # OpenAI chat format
    SELF_INSTRUCT = "self_instruct"  # Self-Instruct format


@dataclass
class AlpacaExample:
    """
    Format Alpaca (Stanford)

    Le format le plus simple et populaire pour instruction tuning.

    Structure:
    - instruction: La tâche à accomplir
    - input: Contexte optionnel (peut être vide)
    - output: Réponse attendue
    """
    instruction: str
    input: str
    output: str

    def format_for_training(self) -> str:
        """
        Formate pour training

        Template:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
        """
        if self.input:
            return f"""### Instruction:
{self.instruction}

### Input:
{self.input}

### Response:
{self.output}"""
        else:
            return f"""### Instruction:
{self.instruction}

### Response:
{self.output}"""

    @staticmethod
    def from_dict(data: Dict) -> 'AlpacaExample':
        """Create from dict"""
        return AlpacaExample(
            instruction=data['instruction'],
            input=data.get('input', ''),
            output=data['output']
        )


@dataclass
class Message:
    """Un message dans une conversation"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ShareGPTExample:
    """
    Format ShareGPT (Conversations multi-turn)

    Utilisé pour conversations réalistes avec plusieurs tours.
    Compatible avec ChatGPT, Claude, etc.

    Structure:
    - conversations: Liste de messages (role + content)
    """
    conversations: List[Message]

    def format_for_training(self) -> str:
        """
        Formate pour training

        Format:
        <|system|>
        {system message}
        <|user|>
        {user message}
        <|assistant|>
        {assistant message}
        ...
        """
        formatted = []

        for msg in self.conversations:
            formatted.append(f"<|{msg.role}|>\n{msg.content}")

        return "\n".join(formatted)

    @staticmethod
    def from_dict(data: Dict) -> 'ShareGPTExample':
        """Create from dict"""
        conversations = [
            Message(role=msg['from'], content=msg['value'])
            for msg in data['conversations']
        ]
        return ShareGPTExample(conversations=conversations)


@dataclass
class OpenAIExample:
    """
    Format OpenAI (Chat Completion API)

    Compatible avec l'API OpenAI.

    Structure:
    - messages: Liste de dicts avec role + content
    """
    messages: List[Dict[str, str]]

    def format_for_training(self) -> str:
        """
        Formate pour training

        Utilise des tokens spéciaux pour marquer les rôles
        """
        formatted = []

        for msg in self.messages:
            role = msg['role']
            content = msg['content']

            if role == "system":
                formatted.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        return "\n".join(formatted)

    @staticmethod
    def from_dict(data: Dict) -> 'OpenAIExample':
        """Create from dict"""
        return OpenAIExample(messages=data['messages'])


class InstructionDatasetFormatter:
    """Formatter universel pour datasets d'instruction"""

    @staticmethod
    def auto_detect_format(example: Dict) -> InstructionFormat:
        """
        Détecte automatiquement le format

        Args:
            example: Un exemple du dataset

        Returns:
            Le format détecté
        """
        # Alpaca: a 'instruction', 'input', 'output'
        if all(k in example for k in ['instruction', 'output']):
            return InstructionFormat.ALPACA

        # ShareGPT: a 'conversations'
        if 'conversations' in example:
            return InstructionFormat.SHAREGPT

        # OpenAI: a 'messages'
        if 'messages' in example:
            return InstructionFormat.OPENAI

        raise ValueError(f"Unknown format: {example.keys()}")

    @staticmethod
    def format_example(
        example: Dict,
        format_type: Optional[InstructionFormat] = None
    ) -> str:
        """
        Formate un exemple selon son type

        Args:
            example: Exemple à formater
            format_type: Type de format (auto-detect si None)

        Returns:
            String formaté pour training
        """
        if format_type is None:
            format_type = InstructionDatasetFormatter.auto_detect_format(example)

        if format_type == InstructionFormat.ALPACA:
            alpaca = AlpacaExample.from_dict(example)
            return alpaca.format_for_training()

        elif format_type == InstructionFormat.SHAREGPT:
            sharegpt = ShareGPTExample.from_dict(example)
            return sharegpt.format_for_training()

        elif format_type == InstructionFormat.OPENAI:
            openai = OpenAIExample.from_dict(example)
            return openai.format_for_training()

        else:
            raise ValueError(f"Unsupported format: {format_type}")


# Exemples de datasets populaires
POPULAR_INSTRUCTION_DATASETS = {
    "Alpaca": {
        "name": "tatsu-lab/alpaca",
        "format": "alpaca",
        "size": "52k",
        "description": "Dataset original Alpaca de Stanford",
        "example": {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1. Eat a balanced diet. 2. Exercise regularly. 3. Get enough sleep."
        }
    },

    "Alpaca GPT-4": {
        "name": "vicgalle/alpaca-gpt4",
        "format": "alpaca",
        "size": "52k",
        "description": "Alpaca dataset généré avec GPT-4 (meilleure qualité)",
        "example": {
            "instruction": "Explain quantum computing in simple terms.",
            "input": "",
            "output": "Quantum computing uses quantum bits (qubits) that can be 0, 1, or both simultaneously..."
        }
    },

    "ShareGPT": {
        "name": "RyokoAI/ShareGPT52K",
        "format": "sharegpt",
        "size": "52k",
        "description": "Conversations réelles avec ChatGPT",
        "example": {
            "conversations": [
                {"from": "human", "value": "How do I learn Python?"},
                {"from": "gpt", "value": "Here are steps to learn Python:\n1. Start with basics..."}
            ]
        }
    },

    "Dolly 15k": {
        "name": "databricks/databricks-dolly-15k",
        "format": "alpaca",
        "size": "15k",
        "description": "Dataset créé par humains (Databricks)",
        "example": {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        }
    },

    "FLAN": {
        "name": "conceptofmind/FLAN_2022",
        "format": "alpaca",
        "size": "1.8M",
        "description": "Collection massive de tâches (Google)",
        "example": {
            "instruction": "Sentiment analysis",
            "input": "This movie was amazing!",
            "output": "Positive"
        }
    }
}


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("INSTRUCTION TUNING FORMATS")
    print("="*60)
    print()

    # Exemple Alpaca
    print("### Format Alpaca")
    alpaca_data = {
        "instruction": "Translate to French",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous ?"
    }
    alpaca = AlpacaExample.from_dict(alpaca_data)
    print(alpaca.format_for_training())
    print()

    print("-"*60)
    print()

    # Exemple ShareGPT
    print("### Format ShareGPT")
    sharegpt_data = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "What is Python?"},
            {"from": "gpt", "value": "Python is a high-level programming language..."}
        ]
    }
    sharegpt = ShareGPTExample.from_dict(sharegpt_data)
    print(sharegpt.format_for_training())
    print()

    print("-"*60)
    print()

    # Exemple OpenAI
    print("### Format OpenAI")
    openai_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain AI in one sentence."},
            {"role": "assistant", "content": "AI is the simulation of human intelligence by machines."}
        ]
    }
    openai = OpenAIExample.from_dict(openai_data)
    print(openai.format_for_training())
    print()

    print("="*60)
    print("POPULAR DATASETS")
    print("="*60)
    print()

    for name, info in list(POPULAR_INSTRUCTION_DATASETS.items())[:3]:
        print(f"### {name}")
        print(f"Dataset: {info['name']}")
        print(f"Format: {info['format']}")
        print(f"Size: {info['size']}")
        print(f"Description: {info['description']}")
        print()

    print("\n💡 Tip: Utilisez Alpaca format pour simplicité, ShareGPT pour conversations multi-turn")
```

## 2. Création de Données d'Instruction

```python
"""
Techniques pour créer des données d'instruction

Méthodes:
1. Self-Instruct (génération avec LLM)
2. Distillation (utiliser GPT-4/Claude)
3. Humain (coûteux mais haute qualité)
4. Augmentation (variations)
"""

from typing import List, Dict
import random


class SelfInstructGenerator:
    """
    Self-Instruct: Génération automatique d'instructions

    Paper: "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
    Wang et al. (2023)

    Principe:
    1. Créer instructions seed (175 exemples manuels)
    2. LLM génère nouvelles instructions similaires
    3. LLM génère input/output pour ces instructions
    4. Filtrer et valider
    5. Répéter
    """

    # Instructions seed (exemples)
    SEED_INSTRUCTIONS = [
        "Write a poem about",
        "Translate the following to French",
        "Summarize this article",
        "Answer the question based on context",
        "Generate a creative story about",
        "Explain the concept of",
        "List the pros and cons of",
        "Compare and contrast",
        "Write code to",
        "Debug the following code"
    ]

    # Domaines (pour diversité)
    DOMAINS = [
        "science", "history", "technology", "literature",
        "mathematics", "cooking", "sports", "music",
        "art", "business", "health", "education"
    ]

    @staticmethod
    def generate_instruction_prompt(
        seed_instructions: List[str],
        num_examples: int = 3
    ) -> str:
        """
        Génère un prompt pour créer de nouvelles instructions

        Args:
            seed_instructions: Instructions existantes
            num_examples: Nombre d'exemples à montrer

        Returns:
            Prompt pour LLM
        """

        # Sample random instructions
        sampled = random.sample(seed_instructions, min(num_examples, len(seed_instructions)))

        prompt = f"""You are an expert at creating instruction-following tasks.

Here are {num_examples} example instructions:

{chr(10).join(f"{i+1}. {inst}" for i, inst in enumerate(sampled))}

Generate 5 new diverse instructions that are different from the examples above.
The instructions should cover a variety of tasks and domains.

Output format (one per line):
- [instruction]
"""

        return prompt

    @staticmethod
    def generate_input_output_prompt(instruction: str) -> str:
        """
        Génère prompt pour créer input/output

        Args:
            instruction: L'instruction

        Returns:
            Prompt pour générer input/output
        """

        prompt = f"""Given the following instruction, generate an appropriate input (if needed) and output.

Instruction: {instruction}

If the instruction doesn't require additional input, leave input empty.

Output in JSON format:
{{
    "instruction": "{instruction}",
    "input": "...",
    "output": "..."
}}
"""

        return prompt

    @staticmethod
    def validate_example(example: Dict) -> bool:
        """
        Valide un exemple généré

        Critères:
        - Output pas trop court (<10 chars)
        - Output pas identique à input
        - Pas de contenu inapproprié (basic check)
        """

        output = example.get('output', '')
        input_text = example.get('input', '')

        # Trop court
        if len(output) < 10:
            return False

        # Output = Input (copie)
        if output == input_text and len(input_text) > 0:
            return False

        # Basic inappropriate content check
        inappropriate = ['hate', 'violence', 'nsfw']
        if any(word in output.lower() for word in inappropriate):
            return False

        return True


class InstructionAugmenter:
    """
    Augmente les données d'instruction existantes

    Techniques:
    1. Paraphrase instructions
    2. Variantes d'input
    3. Back-translation
    4. Template expansion
    """

    @staticmethod
    def paraphrase_instruction(instruction: str) -> List[str]:
        """
        Génère paraphrases d'une instruction

        Returns:
            Liste de paraphrases
        """

        # Templates de paraphrase
        templates = {
            "write": ["compose", "create", "draft", "generate"],
            "explain": ["describe", "clarify", "elaborate on", "detail"],
            "list": ["enumerate", "itemize", "catalog", "outline"],
            "summarize": ["condense", "brief", "abstract", "synopsis of"]
        }

        paraphrases = [instruction]

        # Simple word replacement
        for original, replacements in templates.items():
            if original in instruction.lower():
                for replacement in replacements[:2]:  # Max 2 variants
                    paraphrased = instruction.lower().replace(original, replacement)
                    paraphrases.append(paraphrased.capitalize())

        return paraphrases

    @staticmethod
    def create_variants(
        example: AlpacaExample,
        num_variants: int = 3
    ) -> List[AlpacaExample]:
        """
        Crée variantes d'un exemple

        Args:
            example: Exemple original
            num_variants: Nombre de variantes

        Returns:
            Liste de variantes
        """

        variants = [example]

        # Paraphrase instruction
        paraphrases = InstructionAugmenter.paraphrase_instruction(example.instruction)

        for paraphrase in paraphrases[1:num_variants+1]:
            variant = AlpacaExample(
                instruction=paraphrase,
                input=example.input,
                output=example.output
            )
            variants.append(variant)

        return variants


# Exemple
if __name__ == "__main__":
    print("\n=== Self-Instruct Generation ===\n")

    generator = SelfInstructGenerator()

    # Generate prompt pour nouvelles instructions
    prompt = generator.generate_instruction_prompt(
        generator.SEED_INSTRUCTIONS,
        num_examples=3
    )

    print("Prompt pour générer instructions:")
    print(prompt)

    print("\n" + "="*60 + "\n")

    # Generate prompt pour input/output
    new_instruction = "Explain the benefits of exercise"
    io_prompt = generator.generate_input_output_prompt(new_instruction)

    print("Prompt pour générer input/output:")
    print(io_prompt)

    print("\n" + "="*60 + "\n")

    # Test validation
    examples = [
        {"instruction": "Test", "input": "", "output": "This is a valid output"},
        {"instruction": "Test", "input": "", "output": "Bad"},  # Trop court
        {"instruction": "Test", "input": "Copy me", "output": "Copy me"},  # Identique
    ]

    print("Validation d'exemples:")
    for i, ex in enumerate(examples):
        is_valid = generator.validate_example(ex)
        print(f"  Example {i+1}: {'✅ Valid' if is_valid else '❌ Invalid'}")

    print("\n" + "="*60 + "\n")

    # Test augmentation
    print("=== Instruction Augmentation ===\n")

    augmenter = InstructionAugmenter()

    original = AlpacaExample(
        instruction="Write a poem about nature",
        input="",
        output="The trees sway in the gentle breeze..."
    )

    variants = augmenter.create_variants(original, num_variants=2)

    print(f"Original: {original.instruction}")
    for i, var in enumerate(variants[1:], 1):
        print(f"Variant {i}: {var.instruction}")
```

*[Suite avec techniques avancées et projet dans les parties 2-3...]*

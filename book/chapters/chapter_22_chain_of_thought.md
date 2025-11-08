# Chapitre 22: Chain-of-Thought et Raisonnement Avancé

## Introduction au Raisonnement

Le **raisonnement** est la capacité des LLMs à décomposer des problèmes complexes et à suivre des étapes logiques pour arriver à une solution. Les techniques de Chain-of-Thought permettent d'améliorer drastiquement cette capacité.

```python
"""
REASONING DANS LES LLMs

Problème: Sans guidance, LLMs "sautent" aux conclusions
  Question: "Roger a 5 balles de tennis. Il achète 2 paquets de 3. Combien en a-t-il?"

  Sans CoT:
    LLM → "11 balles"  ❌ (mauvais)

  Avec CoT:
    LLM → "Étape 1: Roger commence avec 5 balles
           Étape 2: 2 paquets × 3 balles = 6 balles
           Étape 3: 5 + 6 = 11 balles
           Réponse: 11"  ✅ (correct ET explicite)

Techniques de reasoning:

1. CHAIN-OF-THOUGHT (CoT) - Wei et al. 2022:
   • Prompt: "Résolvons étape par étape"
   • LLM génère raisonnement intermédiaire
   • Amélioration: +10-50% sur math, logique

2. ZERO-SHOT CoT - Kojima et al. 2022:
   • Juste ajouter "Let's think step by step"
   • Pas d'exemples nécessaires
   • Fonctionne sur GPT-3.5+

3. SELF-CONSISTENCY - Wang et al. 2022:
   • Générer N solutions différentes
   • Choisir réponse majoritaire
   • Amélioration: +15-25% sur CoT simple

4. TREE-OF-THOUGHTS (ToT) - Yao et al. 2023:
   • Explorer arbre de possibilités
   • Backtracking si deadend
   • SOTA pour problèmes complexes

5. PROGRAM-OF-THOUGHTS (PoT) - Chen et al. 2022:
   • Générer code Python au lieu de calculs
   • Executer code pour réponse exacte
   • Parfait pour math

Quand utiliser?
  ✅ Math, logique, puzzles
  ✅ Planification multi-étapes
  ✅ Analyse complexe
  ❌ Questions simples (overhead inutile)
  ❌ Tâches créatives (rigidité)

Performance (GSM8K math benchmark):
  GPT-3 (no CoT): 17%
  GPT-3 + CoT: 58%
  GPT-3 + Self-Consistency: 74%
  GPT-4 + CoT: 95%
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import Counter
import re
import json


# ============================================================================
# CHAIN-OF-THOUGHT (CoT) PROMPTING
# ============================================================================

@dataclass
class CoTExample:
    """Exemple pour few-shot Chain-of-Thought"""
    question: str
    reasoning: str  # Raisonnement étape par étape
    answer: str


class ChainOfThoughtPrompter:
    """
    Prompting Chain-of-Thought (CoT)

    Deux modes:
      1. Few-shot: Fournir exemples avec raisonnement
      2. Zero-shot: "Let's think step by step"

    Paper: Chain-of-Thought Prompting Elicits Reasoning in LLMs
           (Wei et al., NeurIPS 2022)
    """

    def __init__(self, model: Any):
        self.model = model

    def few_shot_cot(
        self,
        question: str,
        examples: List[CoTExample],
        max_tokens: int = 512
    ) -> Tuple[str, str]:
        """
        Few-shot Chain-of-Thought

        Args:
            question: Question à résoudre
            examples: Exemples avec raisonnement
            max_tokens: Tokens max pour génération

        Returns:
            (reasoning, answer)

        Exemple:
            >>> examples = [
            ...     CoTExample(
            ...         question="Roger a 5 balles...",
            ...         reasoning="Étape 1: ...",
            ...         answer="11"
            ...     )
            ... ]
            >>> reasoning, answer = prompter.few_shot_cot(
            ...     "Julie a 10 pommes...",
            ...     examples
            ... )
        """
        print("\n" + "="*80)
        print("FEW-SHOT CHAIN-OF-THOUGHT")
        print("="*80)

        # Construire prompt avec exemples
        prompt_parts = []

        # Ajouter exemples
        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"Q: {ex.question}")
            prompt_parts.append(f"A: {ex.reasoning}")
            prompt_parts.append(f"Donc la réponse est: {ex.answer}\n")

        # Ajouter question cible
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A: ")

        prompt = "\n".join(prompt_parts)

        print(f"Question: {question}")
        print(f"Exemples fournis: {len(examples)}")

        # Générer (simulation)
        # En production: response = self.model.generate(prompt, max_tokens=max_tokens)
        response = """Étape 1: Identifions les quantités initiales
Étape 2: Calculons les changements
Étape 3: Additionnons pour le total
Donc la réponse est: 42"""

        # Parser réponse
        reasoning, answer = self._parse_cot_response(response)

        print(f"\n💭 Raisonnement:")
        print(reasoning)
        print(f"\n✅ Réponse finale: {answer}")

        return reasoning, answer

    def zero_shot_cot(
        self,
        question: str,
        max_tokens: int = 512
    ) -> Tuple[str, str]:
        """
        Zero-shot Chain-of-Thought

        Ajoute simplement "Let's think step by step" au prompt

        Args:
            question: Question à résoudre

        Returns:
            (reasoning, answer)

        Paper: Large Language Models are Zero-Shot Reasoners
               (Kojima et al., NeurIPS 2022)
        """
        print("\n" + "="*80)
        print("ZERO-SHOT CHAIN-OF-THOUGHT")
        print("="*80)
        print(f"Question: {question}")

        # Prompt magique
        prompt = f"{question}\n\nLet's think step by step."

        print(f"Prompt: '{prompt}'")

        # Générer (simulation)
        response = """Étape 1: Analysons le problème
Étape 2: Identifions ce qu'on cherche
Étape 3: Appliquons la logique
Donc la réponse est: 42"""

        reasoning, answer = self._parse_cot_response(response)

        print(f"\n💭 Raisonnement généré:")
        print(reasoning)
        print(f"\n✅ Réponse: {answer}")

        return reasoning, answer

    def _parse_cot_response(self, response: str) -> Tuple[str, str]:
        """
        Parse la réponse CoT pour extraire raisonnement et réponse

        Format attendu:
            Étape 1: ...
            Étape 2: ...
            Donc la réponse est: [ANSWER]
        """
        # Trouver "Donc la réponse est:"
        match = re.search(
            r"(?:Donc |Ainsi |Par conséquent )?(?:la |L')?réponse (?:est|finale est):?\s*(.+)",
            response,
            re.IGNORECASE
        )

        if match:
            answer = match.group(1).strip()
            reasoning = response[:match.start()].strip()
        else:
            # Fallback: dernière ligne = réponse
            lines = response.strip().split('\n')
            answer = lines[-1] if lines else ""
            reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else ""

        return reasoning, answer


# ============================================================================
# SELF-CONSISTENCY
# ============================================================================

"""
SELF-CONSISTENCY = Générer plusieurs chemins de raisonnement

Algorithme:
  1. Générer N solutions indépendantes (avec CoT)
  2. Extraire réponse finale de chaque solution
  3. Choisir réponse majoritaire (vote)

Exemple:
  Question: "Un restaurant a 23 tables. Chaque table a 4 chaises. Combien de chaises?"

  Path 1: "23 × 4 = 92 chaises" ✓
  Path 2: "4 tables × 23 chaises... non wait, 23 × 4 = 92" ✓
  Path 3: "23 + 4 = 27... " ✗
  Path 4: "23 × 4 = 92" ✓
  Path 5: "23 tables, 4 per table = 92" ✓

  Vote: 92 (4/5) → Réponse: 92 ✅

Avantages:
  ✅ Robuste aux erreurs ponctuelles
  ✅ +15-25% accuracy vs CoT simple
  ✅ Fonctionne mieux avec plus de paths

Coût:
  ❌ N fois plus de tokens (ex: 10x si N=10)
  ❌ N appels au LLM

Paper: Self-Consistency Improves Chain of Thought Reasoning
       (Wang et al., ICLR 2023)
"""

class SelfConsistency:
    """
    Self-Consistency avec Chain-of-Thought

    Génère plusieurs raisonnements indépendants et vote
    """

    def __init__(
        self,
        cot_prompter: ChainOfThoughtPrompter,
        num_paths: int = 5
    ):
        """
        Args:
            cot_prompter: CoT prompter à utiliser
            num_paths: Nombre de chemins de raisonnement
        """
        self.cot_prompter = cot_prompter
        self.num_paths = num_paths

    def solve_with_consistency(
        self,
        question: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Résout avec self-consistency

        Args:
            question: Question à résoudre
            temperature: > 0 pour diversité des paths

        Returns:
            {
                "final_answer": str,
                "paths": List[Dict],
                "votes": Dict[str, int]
            }
        """
        print("\n" + "="*80)
        print("SELF-CONSISTENCY")
        print("="*80)
        print(f"Question: {question}")
        print(f"Nombre de paths: {self.num_paths}")
        print(f"Temperature: {temperature}")

        # Générer N paths indépendants
        paths = []

        for i in range(self.num_paths):
            print(f"\n--- Path {i+1}/{self.num_paths} ---")

            # Générer avec CoT (zero-shot pour diversité)
            reasoning, answer = self.cot_prompter.zero_shot_cot(question)

            paths.append({
                "path_id": i + 1,
                "reasoning": reasoning,
                "answer": self._normalize_answer(answer)
            })

        # Voter
        answers = [p["answer"] for p in paths]
        vote_counts = Counter(answers)

        print(f"\n📊 VOTES:")
        for answer, count in vote_counts.most_common():
            print(f"  '{answer}': {count}/{self.num_paths} ({count/self.num_paths:.0%})")

        # Réponse majoritaire
        final_answer, max_votes = vote_counts.most_common(1)[0]

        print(f"\n✅ RÉPONSE FINALE (majorité): '{final_answer}'")
        print(f"   Confiance: {max_votes}/{self.num_paths} ({max_votes/self.num_paths:.0%})")

        return {
            "final_answer": final_answer,
            "paths": paths,
            "votes": dict(vote_counts),
            "confidence": max_votes / self.num_paths
        }

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalise une réponse pour le vote

        Exemples:
          "92 chaises" → "92"
          "La réponse est 92" → "92"
          "ninety-two" → "92"
        """
        # Extraire nombres
        numbers = re.findall(r'\d+\.?\d*', answer)

        if numbers:
            return numbers[0]

        # Sinon, nettoyer et lowercase
        cleaned = answer.strip().lower()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)

        return cleaned


# ============================================================================
# PROGRAM-OF-THOUGHTS (PoT)
# ============================================================================

"""
PROGRAM-OF-THOUGHTS = Générer code au lieu de calculs

Idée: Pour problèmes math/logique, générer Python code

Exemple:
  Question: "Si x² + 5x + 6 = 0, quelles sont les valeurs de x?"

  CoT traditionnel:
    "Utilisons la formule quadratique: x = (-b ± √(b²-4ac)) / 2a
     a=1, b=5, c=6
     x = (-5 ± √(25-24)) / 2 = (-5 ± 1) / 2
     x = -2 ou x = -3"
    → Risque d'erreurs de calcul

  PoT:
    ```python
    import numpy as np
    coeffs = [1, 5, 6]
    roots = np.roots(coeffs)
    print(roots)
    ```
    → Execute: [-3. -2.]
    → Exact, pas d'erreurs

Avantages:
  ✅ Calculs exacts (pas d'erreurs arithmétiques)
  ✅ Peut utiliser libs (numpy, sympy, etc.)
  ✅ SOTA sur math benchmarks

Limitations:
  ❌ Nécessite execution sandbox
  ❌ Pas bon pour raisonnement qualitatif

Paper: Program of Thoughts Prompting (Chen et al., 2022)
"""

class ProgramOfThoughts:
    """
    Program-of-Thoughts (PoT)

    Génère et execute du code Python pour résoudre problèmes
    """

    def __init__(self, model: Any):
        self.model = model

    def solve_with_code(
        self,
        question: str,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Résout en générant code Python

        Args:
            question: Question (math, logique, etc.)

        Returns:
            {
                "code": str,
                "output": str,
                "answer": Any
            }
        """
        print("\n" + "="*80)
        print("PROGRAM-OF-THOUGHTS")
        print("="*80)
        print(f"Question: {question}")

        # Prompt pour générer code
        prompt = f"""Résous ce problème en écrivant du code Python.

Question: {question}

Code Python:
```python"""

        # Générer code (simulation)
        # En production: code = self.model.generate(prompt)
        code = """import numpy as np

# Résolution de l'équation x² + 5x + 6 = 0
coefficients = [1, 5, 6]
roots = np.roots(coefficients)

print(f"Les racines sont: {roots}")
answer = list(roots)
"""

        print(f"\n💻 Code généré:")
        print(code)

        # Executer code
        output, answer = self._execute_code(code)

        print(f"\n📤 Output:")
        print(output)

        print(f"\n✅ Réponse: {answer}")

        return {
            "code": code,
            "output": output,
            "answer": answer
        }

    def _execute_code(self, code: str) -> Tuple[str, Any]:
        """
        Execute code Python de manière sécurisée

        ATTENTION: En production, utiliser sandbox (Docker, pyodide, etc.)

        Args:
            code: Code Python à executer

        Returns:
            (stdout, answer_variable)
        """
        # En production: utiliser sandbox sécurisé
        # Exemple avec RestrictedPython, ou Docker container

        # Simulation pour démo
        output = "Les racines sont: [-3. -2.]"
        answer = [-3.0, -2.0]

        return output, answer


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_chain_of_thought():
    """Démo Chain-of-Thought"""
    print("="*80)
    print("DÉMONSTRATION: CHAIN-OF-THOUGHT")
    print("="*80)

    prompter = ChainOfThoughtPrompter(model=None)

    # Test 1: Zero-shot CoT
    print("\n1️⃣  ZERO-SHOT CoT")
    print("-" * 80)

    question = """Un magasin vend des pommes à 2€ le kilo.
Marie achète 3.5 kilos et donne un billet de 20€.
Combien de monnaie recevra-t-elle?"""

    reasoning, answer = prompter.zero_shot_cot(question)

    # Test 2: Few-shot CoT
    print("\n\n2️⃣  FEW-SHOT CoT")
    print("-" * 80)

    examples = [
        CoTExample(
            question="Roger a 5 balles de tennis. Il achète 2 paquets de 3 balles. Combien en a-t-il?",
            reasoning="""Étape 1: Roger commence avec 5 balles
Étape 2: Chaque paquet contient 3 balles, donc 2 paquets = 2 × 3 = 6 balles
Étape 3: Total = balles initiales + nouvelles balles = 5 + 6 = 11""",
            answer="11 balles"
        )
    ]

    question2 = "Sophie a 8 billes. Elle en perd 3 puis en gagne 7. Combien en a-t-elle?"

    reasoning, answer = prompter.few_shot_cot(question2, examples)


def demo_self_consistency():
    """Démo Self-Consistency"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: SELF-CONSISTENCY")
    print("="*80)

    prompter = ChainOfThoughtPrompter(model=None)
    sc = SelfConsistency(prompter, num_paths=5)

    question = """Une salle de classe a 6 rangées de bureaux.
Chaque rangée a 5 bureaux.
Si chaque bureau peut accueillir 2 élèves, combien d'élèves au total?"""

    result = sc.solve_with_consistency(question, temperature=0.7)

    print(f"\n📊 Résumé:")
    print(f"  Paths générés: {len(result['paths'])}")
    print(f"  Réponse finale: {result['final_answer']}")
    print(f"  Confiance: {result['confidence']:.0%}")


def demo_program_of_thoughts():
    """Démo Program-of-Thoughts"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: PROGRAM-OF-THOUGHTS")
    print("="*80)

    pot = ProgramOfThoughts(model=None)

    question = """Résous l'équation quadratique: x² + 5x + 6 = 0
Quelles sont les valeurs de x?"""

    result = pot.solve_with_code(question)


def compare_techniques():
    """Compare les différentes techniques"""
    print("\n\n" + "="*80)
    print("COMPARAISON DES TECHNIQUES")
    print("="*80)

    print("""
┌────────────────────┬──────────────┬─────────────┬──────────────┬─────────────┐
│ Technique          │ Accuracy     │ Coût        │ Latence      │ Use Case    │
├────────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
│ Direct (no CoT)    │ ⭐ Faible    │ ⭐ Min      │ ⭐ Rapide    │ Questions   │
│                    │              │             │              │ simples     │
├────────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
│ Zero-Shot CoT      │ ⭐⭐⭐ Bon   │ ⭐⭐ Moyen  │ ⭐⭐ OK      │ General     │
│                    │              │             │              │ reasoning   │
├────────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
│ Few-Shot CoT       │ ⭐⭐⭐⭐ Très│ ⭐⭐ Moyen  │ ⭐⭐ OK      │ Domain-     │
│                    │ bon          │             │              │ specific    │
├────────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
│ Self-Consistency   │ ⭐⭐⭐⭐⭐   │ ⭐ Élevé    │ ⭐ Lent      │ Critical    │
│ (N=5-10)           │ Excellent    │ (Nx cost)   │ (N appels)   │ decisions   │
├────────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
│ Program-of-        │ ⭐⭐⭐⭐⭐   │ ⭐⭐ Moyen  │ ⭐⭐⭐ Bon   │ Math,       │
│ Thoughts           │ Exact (math) │             │              │ logic       │
└────────────────────┴──────────────┴─────────────┴──────────────┴─────────────┘

BENCHMARK: GSM8K (8th grade math, 1319 questions)

Model: GPT-3 (175B)
  • Direct: 17.7%
  • Zero-Shot CoT: 40.7% (+23%)
  • Few-Shot CoT (8 examples): 58.1% (+40%)
  • Self-Consistency (CoT + N=40): 74.4% (+57%)

Model: GPT-4
  • Direct: 80.8%
  • Zero-Shot CoT: 87.1%
  • Few-Shot CoT: 92.0%
  • Self-Consistency: 95.2%

RECOMMANDATIONS:

1. Questions simples:
   → Direct (pas de CoT nécessaire)

2. Math/logique standard:
   → Zero-Shot CoT ("Let's think step by step")

3. Domain spécifique:
   → Few-Shot CoT avec exemples

4. Décisions critiques:
   → Self-Consistency (N=5-10)

5. Math complexe:
   → Program-of-Thoughts

6. Combo optimal (accuracy max):
   → Self-Consistency + Program-of-Thoughts
    """)


if __name__ == "__main__":
    # Démos
    demo_chain_of_thought()
    demo_self_consistency()
    demo_program_of_thoughts()

    # Comparaison
    compare_techniques()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 1")
    print("="*80)
    print("""
1. CHAIN-OF-THOUGHT (CoT)
   • "Let's think step by step" = +20-40% accuracy
   • Décompose problèmes complexes
   • Deux modes: Zero-shot, Few-shot

2. SELF-CONSISTENCY
   • Génère N paths indépendants
   • Vote majoritaire
   • +15-25% vs CoT simple
   • Coût: N fois plus de tokens

3. PROGRAM-OF-THOUGHTS
   • Génère code au lieu de calculs
   • Calculs exacts (pas d'erreurs arithmétiques)
   • SOTA pour math

4. QUAND UTILISER?
   Simple question → Direct
   Math/logique → Zero-Shot CoT
   Domain spécifique → Few-Shot CoT
   Critique → Self-Consistency
   Math complexe → PoT

PROCHAINE PARTIE: Tree-of-Thoughts et techniques avancées
    """)

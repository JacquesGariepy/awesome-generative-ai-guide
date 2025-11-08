# Chapitre 23 - Partie 2: Projets Avancés

## Projets 3-8: Applications Spécialisées

Cette partie couvre des projets plus techniques intégrant RAG, agents, multimodal et reasoning avancé.

```python
"""
PROJETS AVANCÉS - PARTIE 2

Projets couverts:
  3. Code Review Automatique avec Agents
  4. Assistant Médical RAG
  5. Analyseur Financier Multi-Sources
  6. Générateur de Tests Automatiques
  7. Traducteur Technique Contextualisé
  8. Système de Veille Technologique

Techniques intégrées:
  • Multi-agents (coordination)
  • RAG avancé (hybrid search)
  • Multimodal (code, images, PDFs)
  • Long context (documents entiers)
  • Chain-of-Thought (analyse complexe)
  • Fine-tuning domaine spécifique
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


# ============================================================================
# PROJET 3: CODE REVIEW AUTOMATIQUE
# ============================================================================

"""
PROJET 3: Automated Code Review System

Cas d'usage:
  • Review automatique des Pull Requests
  • Détection de bugs potentiels
  • Suggestions d'optimisation
  • Vérification conformité standards
  • Génération de tests manquants

Stack:
  • Fine-tuned CodeLlama sur code reviews
  • Multi-agents: Security, Performance, Style, Tests
  • Static analysis integration (AST parsing)
  • Git integration (GitHub/GitLab)

Architecture Multi-Agents:
  Manager Agent
    ├─ Security Agent: Vulnérabilités
    ├─ Performance Agent: Optimisations
    ├─ Style Agent: Standards de code
    └─ Test Agent: Coverage et tests manquants

Flow:
  1. PR créée → Webhook
  2. Fetch code diff
  3. Parse AST (Abstract Syntax Tree)
  4. Distribute aux agents spécialisés
  5. Agrégation des reviews
  6. Post comment sur PR
  7. Auto-approve si score > seuil

ROI:
  • -70% temps de review humain
  • +40% bugs détectés avant merge
  • Standards de code uniformes
"""

class ReviewSeverity(Enum):
    """Sévérité d'un commentaire de review"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ReviewComment:
    """Commentaire de code review"""
    file_path: str
    line_number: int
    severity: ReviewSeverity
    category: str  # "security", "performance", "style", "tests"
    message: str
    suggestion: Optional[str] = None
    agent: Optional[str] = None


class CodeReviewAgent:
    """Agent spécialisé pour code review"""

    def __init__(
        self,
        name: str,
        specialty: str,
        model: Any
    ):
        self.name = name
        self.specialty = specialty
        self.model = model

    def review_file(
        self,
        file_path: str,
        code: str,
        diff: Optional[str] = None
    ) -> List[ReviewComment]:
        """
        Review un fichier selon la spécialité

        Args:
            file_path: Chemin du fichier
            code: Code source complet
            diff: Diff Git (lignes modifiées)

        Returns:
            Liste de commentaires
        """
        print(f"\n🔍 {self.name} reviewing {file_path}")

        comments = []

        if self.specialty == "security":
            comments = self._check_security(file_path, code)
        elif self.specialty == "performance":
            comments = self._check_performance(file_path, code)
        elif self.specialty == "style":
            comments = self._check_style(file_path, code)
        elif self.specialty == "tests":
            comments = self._check_tests(file_path, code)

        print(f"  → {len(comments)} commentaires")

        return comments

    def _check_security(self, file_path: str, code: str) -> List[ReviewComment]:
        """Vérifie sécurité"""
        comments = []

        # Patterns de sécurité courants
        security_patterns = {
            "sql_injection": r"execute\([\"'].*\+.*[\"']\)",
            "hardcoded_secret": r"(password|secret|key)\s*=\s*[\"'][^\"']{8,}[\"']",
            "eval_usage": r"eval\(",
            "pickle_unsafe": r"pickle\.loads\(",
        }

        for issue, pattern in security_patterns.items():
            import re
            if re.search(pattern, code, re.IGNORECASE):
                comments.append(ReviewComment(
                    file_path=file_path,
                    line_number=0,  # Simplification
                    severity=ReviewSeverity.CRITICAL,
                    category="security",
                    message=f"Vulnérabilité potentielle: {issue}",
                    suggestion="Utiliser des requêtes paramétrées / éviter eval / etc.",
                    agent=self.name
                ))

        return comments

    def _check_performance(self, file_path: str, code: str) -> List[ReviewComment]:
        """Vérifie performance"""
        comments = []

        # Patterns de performance
        if "for " in code and " in " in code and ".append(" in code:
            comments.append(ReviewComment(
                file_path=file_path,
                line_number=0,
                severity=ReviewSeverity.WARNING,
                category="performance",
                message="List comprehension plus efficace que for loop + append",
                suggestion="Utiliser [x for x in items] au lieu de for x in items: list.append(x)",
                agent=self.name
            ))

        return comments

    def _check_style(self, file_path: str, code: str) -> List[ReviewComment]:
        """Vérifie style"""
        comments = []

        # PEP 8 checks (simplifié)
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Ligne trop longue
            if len(line) > 88:  # Black default
                comments.append(ReviewComment(
                    file_path=file_path,
                    line_number=i,
                    severity=ReviewSeverity.INFO,
                    category="style",
                    message=f"Ligne trop longue ({len(line)} caractères > 88)",
                    agent=self.name
                ))

        return comments

    def _check_tests(self, file_path: str, code: str) -> List[ReviewComment]:
        """Vérifie tests"""
        comments = []

        # Vérifier si fichier de test existe
        if not file_path.startswith("test_") and "tests/" not in file_path:
            # Code de prod sans tests?
            if "def " in code or "class " in code:
                comments.append(ReviewComment(
                    file_path=file_path,
                    line_number=0,
                    severity=ReviewSeverity.WARNING,
                    category="tests",
                    message="Aucun test détecté pour ce fichier",
                    suggestion=f"Créer tests/{file_path} avec tests unitaires",
                    agent=self.name
                ))

        return comments


class MultiAgentCodeReviewer:
    """
    Système de review multi-agents

    Coordonne plusieurs agents spécialisés
    """

    def __init__(self, model: Any):
        # Créer agents spécialisés
        self.agents = [
            CodeReviewAgent("SecurityBot", "security", model),
            CodeReviewAgent("PerfBot", "performance", model),
            CodeReviewAgent("StyleBot", "style", model),
            CodeReviewAgent("TestBot", "tests", model)
        ]

    def review_pull_request(
        self,
        files: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Review complet d'une Pull Request

        Args:
            files: Liste de {path, content, diff}

        Returns:
            {
                "comments": List[ReviewComment],
                "score": float,
                "auto_approve": bool,
                "summary": str
            }
        """
        print("\n" + "="*80)
        print("CODE REVIEW MULTI-AGENTS")
        print("="*80)
        print(f"Fichiers à reviewer: {len(files)}")

        all_comments = []

        # Review par chaque agent
        for file_info in files:
            print(f"\n📄 Fichier: {file_info['path']}")

            for agent in self.agents:
                comments = agent.review_file(
                    file_path=file_info['path'],
                    code=file_info['content'],
                    diff=file_info.get('diff')
                )
                all_comments.extend(comments)

        # Agréger et scorer
        score = self._calculate_score(all_comments)
        auto_approve = score >= 0.85  # Seuil

        # Générer summary
        summary = self._generate_summary(all_comments, score)

        print(f"\n📊 RÉSULTAT REVIEW")
        print(f"  Commentaires: {len(all_comments)}")
        print(f"  Score: {score:.1%}")
        print(f"  Auto-approve: {'✅ Oui' if auto_approve else '❌ Non'}")

        return {
            "comments": all_comments,
            "score": score,
            "auto_approve": auto_approve,
            "summary": summary
        }

    def _calculate_score(self, comments: List[ReviewComment]) -> float:
        """Calcule score global de la PR"""
        if not comments:
            return 1.0  # Parfait

        # Pénalités selon sévérité
        penalties = {
            ReviewSeverity.CRITICAL: 0.5,
            ReviewSeverity.ERROR: 0.2,
            ReviewSeverity.WARNING: 0.05,
            ReviewSeverity.INFO: 0.01
        }

        total_penalty = sum(
            penalties.get(c.severity, 0) for c in comments
        )

        score = max(0.0, 1.0 - total_penalty)

        return score

    def _generate_summary(
        self,
        comments: List[ReviewComment],
        score: float
    ) -> str:
        """Génère résumé du review"""

        # Grouper par catégorie
        by_category = {}
        for comment in comments:
            cat = comment.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(comment)

        # Compter par sévérité
        by_severity = {}
        for comment in comments:
            sev = comment.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        summary = f"""## Code Review Summary

**Score Global**: {score:.1%}

**Issues par Catégorie**:
"""
        for cat, cat_comments in by_category.items():
            summary += f"\n- **{cat.title()}**: {len(cat_comments)} issues"

        summary += f"""

**Sévérité**:
"""
        for sev, count in sorted(by_severity.items()):
            summary += f"\n- {sev.upper()}: {count}"

        if score >= 0.85:
            summary += "\n\n✅ **Cette PR peut être auto-approuvée**"
        else:
            summary += "\n\n⚠️  **Review humain recommandé**"

        return summary


# ============================================================================
# PROJET 4: ASSISTANT MÉDICAL RAG
# ============================================================================

"""
PROJET 4: Medical Assistant with RAG

Cas d'usage:
  • Assistance aux professionnels de santé
  • Recherche dans littérature médicale
  • Suggestions diagnostiques
  • Recommandations de traitement
  • Références aux guidelines

⚠️  DISCLAIMER: Assistant seulement, pas de diagnostic autonome!

Stack:
  • Fine-tuned medical LLM (BioGPT, Med-PaLM)
  • RAG: PubMed, medical guidelines, drug database
  • Vector DB: Spécialisé medical embeddings
  • Citation tracking (traçabilité)
  • Safety filters (disclaimers, escalation)

Features:
  ✅ Recherche symptômes
  ✅ Suggestions différentielles
  ✅ Drug interactions check
  ✅ Guidelines lookup
  ✅ Citations automatiques
  ✅ Multilingual support

Sécurité:
  ⚠️  Toujours citer sources
  ⚠️  Disclaimer sur chaque réponse
  ⚠️  Escalation si cas complexe
  ⚠️  Logging audit complet
"""

@dataclass
class MedicalQuery:
    """Requête médicale"""
    patient_id: str
    symptoms: List[str]
    medical_history: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    age: Optional[int] = None
    sex: Optional[str] = None


@dataclass
class MedicalResponse:
    """Réponse médicale avec citations"""
    differential_diagnosis: List[str]
    recommended_tests: List[str]
    treatment_suggestions: List[str]
    citations: List[Dict[str, str]]
    confidence: float
    disclaimer: str


class MedicalRAGAssistant:
    """
    Assistant médical avec RAG

    ⚠️  Usage: Assistance professionnels, PAS diagnostic autonome
    """

    def __init__(
        self,
        llm: Any,
        medical_knowledge_base: Any,  # RAG system
        drug_database: Any
    ):
        self.llm = llm
        self.knowledge_base = medical_knowledge_base
        self.drug_db = drug_database

        # Disclaimer obligatoire
        self.DISCLAIMER = """
⚠️  IMPORTANT: Cette information est fournie à titre éducatif seulement.
Elle ne remplace pas l'avis d'un professionnel de santé qualifié.
Consultez toujours un médecin pour diagnostic et traitement.
        """

    def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """
        Traite requête médicale

        Args:
            query: Informations patient et symptômes

        Returns:
            Réponse avec suggestions et citations
        """
        print("\n" + "="*80)
        print("ASSISTANT MÉDICAL RAG")
        print("="*80)
        print(f"Patient: {query.patient_id}")
        print(f"Symptômes: {', '.join(query.symptoms)}")

        # 1. Recherche dans knowledge base
        search_query = self._build_search_query(query)
        relevant_docs = self.knowledge_base.search(
            search_query,
            top_k=10,
            filters={"type": "peer_reviewed"}  # Seulement sources fiables
        )

        print(f"\n📚 Documents médicaux trouvés: {len(relevant_docs)}")

        # 2. Vérifier interactions médicamenteuses
        drug_interactions = []
        if query.current_medications:
            drug_interactions = self._check_drug_interactions(
                query.current_medications
            )

        # 3. Générer diagnostic différentiel
        differential = self._generate_differential(
            query=query,
            context=relevant_docs
        )

        print(f"\n🔬 Diagnostic différentiel: {len(differential)} possibilités")

        # 4. Recommander tests
        tests = self._recommend_tests(query, differential)

        # 5. Suggestions de traitement
        treatments = self._suggest_treatments(
            query=query,
            differential=differential,
            context=relevant_docs,
            drug_interactions=drug_interactions
        )

        # 6. Extraire citations
        citations = self._extract_citations(relevant_docs)

        # 7. Calculer confiance
        confidence = self._calculate_confidence(
            num_sources=len(relevant_docs),
            query_specificity=len(query.symptoms)
        )

        print(f"\n📊 Confiance: {confidence:.1%}")

        return MedicalResponse(
            differential_diagnosis=differential,
            recommended_tests=tests,
            treatment_suggestions=treatments,
            citations=citations,
            confidence=confidence,
            disclaimer=self.DISCLAIMER
        )

    def _build_search_query(self, query: MedicalQuery) -> str:
        """Construit requête de recherche optimisée"""
        parts = []

        # Symptômes
        parts.append("Symptoms: " + ", ".join(query.symptoms))

        # Contexte patient
        if query.age:
            parts.append(f"Age: {query.age}")
        if query.sex:
            parts.append(f"Sex: {query.sex}")
        if query.medical_history:
            parts.append("History: " + ", ".join(query.medical_history))

        return " | ".join(parts)

    def _check_drug_interactions(
        self,
        medications: List[str]
    ) -> List[Dict]:
        """Vérifie interactions médicamenteuses"""
        interactions = []

        # En production: requête DB interactions
        # Simulation
        if len(medications) >= 2:
            interactions.append({
                "drugs": medications[:2],
                "severity": "moderate",
                "description": "Interaction possible, surveillance recommandée"
            })

        return interactions

    def _generate_differential(
        self,
        query: MedicalQuery,
        context: List[str]
    ) -> List[str]:
        """Génère diagnostic différentiel"""

        prompt = f"""Basé sur ces symptômes et informations:

Symptômes: {', '.join(query.symptoms)}
Âge: {query.age or 'N/A'}
Historique: {query.medical_history or 'N/A'}

Contexte médical:
{chr(10).join(context[:3])}

Liste les diagnostics différentiels possibles par ordre de probabilité.
Fournis 3-5 possibilités avec justifications."""

        # En production: LLM generate
        # differential = self.llm.generate(prompt)

        # Simulation
        differential = [
            "Diagnostic 1 (plus probable)",
            "Diagnostic 2 (possible)",
            "Diagnostic 3 (à exclure)"
        ]

        return differential

    def _recommend_tests(
        self,
        query: MedicalQuery,
        differential: List[str]
    ) -> List[str]:
        """Recommande tests diagnostiques"""

        # En production: basé sur guidelines
        tests = [
            "Examen clinique complet",
            "Tests sanguins standard (NFS, CRP)",
            "Imagerie si indiqué"
        ]

        return tests

    def _suggest_treatments(
        self,
        query: MedicalQuery,
        differential: List[str],
        context: List[str],
        drug_interactions: List[Dict]
    ) -> List[str]:
        """Suggère options de traitement"""

        treatments = [
            "Traitement symptomatique initial",
            "Surveillance clinique",
            "Référence spécialiste si nécessaire"
        ]

        # Avertissement si interactions
        if drug_interactions:
            treatments.insert(0, "⚠️  Attention: Interactions médicamenteuses détectées")

        return treatments

    def _extract_citations(self, docs: List[str]) -> List[Dict]:
        """Extrait citations des sources"""
        citations = []

        for i, doc in enumerate(docs[:5]):  # Top 5
            # En production: parser métadonnées réelles
            citations.append({
                "id": f"ref_{i+1}",
                "title": f"Medical Source {i+1}",
                "journal": "PubMed",
                "year": "2024",
                "excerpt": doc[:100] + "..."
            })

        return citations

    def _calculate_confidence(
        self,
        num_sources: int,
        query_specificity: int
    ) -> float:
        """Calcule score de confiance"""

        # Plus de sources = plus confiance
        source_score = min(1.0, num_sources / 10)

        # Plus de symptômes spécifiques = plus confiance
        specificity_score = min(1.0, query_specificity / 5)

        confidence = (source_score + specificity_score) / 2

        return confidence


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_code_review():
    """Démo code review"""
    print("="*80)
    print("PROJET 3: CODE REVIEW AUTOMATIQUE")
    print("="*80)

    reviewer = MultiAgentCodeReviewer(model=None)

    # Fichiers simulés
    files = [
        {
            "path": "src/api/users.py",
            "content": """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    result = execute(query)
    return result

def update_password(user_id, password):
    secret_key = "hardcoded_secret_123456"
    # Update password logic
    pass
""",
            "diff": "+def get_user..."
        },
        {
            "path": "src/utils/helpers.py",
            "content": """
def process_items(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result

def very_long_function_name_that_exceeds_the_maximum_line_length_allowed_by_pep8_standards():
    pass
""",
            "diff": "+def process_items..."
        }
    ]

    # Review
    result = reviewer.review_pull_request(files)

    # Afficher résultats
    print(f"\n\n{result['summary']}")

    print(f"\n\n📝 DÉTAIL DES COMMENTAIRES")
    print("-" * 80)

    for comment in result['comments'][:5]:  # Top 5
        icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "ℹ️"}
        print(f"\n{icon.get(comment.severity.value, '•')} [{comment.severity.value.upper()}] {comment.category}")
        print(f"   Fichier: {comment.file_path}")
        print(f"   Message: {comment.message}")
        if comment.suggestion:
            print(f"   💡 Suggestion: {comment.suggestion}")


def demo_medical_assistant():
    """Démo assistant médical"""
    print("\n\n" + "="*80)
    print("PROJET 4: ASSISTANT MÉDICAL RAG")
    print("="*80)

    # Mock dependencies
    class MockMedicalKB:
        def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[str]:
            return [
                "Study on differential diagnosis of chest pain (2023)",
                "Guidelines for acute coronary syndrome (AHA 2024)",
                "Meta-analysis: Risk factors for cardiac events"
            ]

    assistant = MedicalRAGAssistant(
        llm=None,
        medical_knowledge_base=MockMedicalKB(),
        drug_database=None
    )

    # Requête
    query = MedicalQuery(
        patient_id="PT-12345",
        symptoms=["chest pain", "shortness of breath", "fatigue"],
        age=55,
        sex="M",
        medical_history=["hypertension", "type 2 diabetes"],
        current_medications=["metformin", "lisinopril"]
    )

    # Traiter
    response = assistant.process_query(query)

    # Afficher
    print(f"\n\n📋 RÉPONSE")
    print("-" * 80)

    print(f"\n🔬 Diagnostic Différentiel:")
    for i, diag in enumerate(response.differential_diagnosis, 1):
        print(f"  {i}. {diag}")

    print(f"\n🧪 Tests Recommandés:")
    for test in response.recommended_tests:
        print(f"  • {test}")

    print(f"\n💊 Suggestions de Traitement:")
    for treatment in response.treatment_suggestions:
        print(f"  • {treatment}")

    print(f"\n📚 Citations ({len(response.citations)}):")
    for cit in response.citations[:3]:
        print(f"  [{cit['id']}] {cit['title']} ({cit['year']})")

    print(f"\n{response.disclaimer}")


if __name__ == "__main__":
    demo_code_review()
    demo_medical_assistant()

    print("\n\n" + "="*80)
    print("SUITE DES PROJETS")
    print("="*80)
    print("""
✅ Projets 1-2: Customer Support + Content Marketing
✅ Projets 3-4: Code Review + Medical Assistant

À venir dans Partie 3:
  • Projet 5: Analyseur Financier Multi-Sources
  • Projet 6: Générateur de Tests Automatiques
  • Projet 7-8: Traduction Technique + Veille Techno

À venir dans Partie 4:
  • Projets 9-15: Applications métier diverses

Ensuite:
  • Chapitre 24: Projet Capstone End-to-End
  • Chapitre 25: Best Practices et Production Patterns
    """)

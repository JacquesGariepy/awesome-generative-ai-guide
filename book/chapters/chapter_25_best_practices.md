# Chapitre 25: Best Practices et Production Patterns

## Introduction

Ce chapitre final synthétise les **best practices** essentielles pour déployer et maintenir des systèmes LLM en production à grande échelle.

```python
"""
BEST PRACTICES LLM PRODUCTION - GUIDE COMPLET

Catégories:
  1. Architecture et Design
  2. Performance et Scaling
  3. Sécurité et Privacy
  4. Cost Optimization
  5. Monitoring et Observabilité
  6. Testing et QA
  7. MLOps et CI/CD
  8. User Experience

Ce chapitre couvre:
  ✅ Patterns éprouvés
  ✅ Pièges à éviter
  ✅ Checklists production
  ✅ Troubleshooting guides
  ✅ Decision frameworks
"""


# ============================================================================
# 1. ARCHITECTURE ET DESIGN
# ============================================================================

"""
ARCHITECTURE PATTERNS

Pattern 1: API Gateway + Microservices
  ┌─────────┐
  │ Gateway │ → Auth, Rate Limit, Routing
  └────┬────┘
       ├→ LLM Service (vLLM backend)
       ├→ RAG Service (Vector DB)
       ├→ Agent Service (Orchestration)
       └→ Embedding Service

✅ Avantages:
  • Scalabilité indépendante
  • Failure isolation
  • Technology flexibility

❌ Complexité:
  • Network overhead
  • Distributed tracing required

Pattern 2: Monolith Modulaire
  ┌──────────────────┐
  │   Single App     │
  │  ├─ LLM Module   │
  │  ├─ RAG Module   │
  │  └─ Agents       │
  └──────────────────┘

✅ Avantages:
  • Simplicité deployment
  • Latence minimale
  • Facile à debug

❌ Limitations:
  • Scaling moins flexible
  • Deployment all-or-nothing

Recommandation:
  • Commencer monolithe modulaire
  • Migrer vers microservices si >100k req/jour
  • Extraire services critiques d'abord (LLM inference)
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ArchitectureDecisionRecord:
    """
    Architecture Decision Record (ADR)

    Documenter décisions importantes
    """
    id: str
    title: str
    date: str
    status: str  # "proposed", "accepted", "deprecated", "superseded"
    context: str
    decision: str
    consequences: str
    alternatives_considered: List[str]


# Exemple ADR
ADR_001 = ArchitectureDecisionRecord(
    id="ADR-001",
    title="Choix de vLLM comme moteur d'inférence",
    date="2024-01-15",
    status="accepted",
    context="""
Besoin: Servir LLaMA 70B avec latence <500ms P95 et throughput >100 req/s

Options évaluées:
- HuggingFace Transformers: Simple mais lent
- TensorRT-LLM: Rapide mais complexe
- vLLM: Bon compromis
""",
    decision="""
Adopter vLLM comme moteur principal d'inférence.

Raisons:
1. PagedAttention réduit mémoire de 50%
2. Continuous batching → +2x throughput
3. Production-ready (OpenAI l'utilise)
4. Support multi-GPU natif
""",
    consequences="""
Positif:
+ Atteinte objectifs perf
+ Communauté active
+ Intégration OpenAI API facile

Négatif:
- Dépendance externe
- Nécessite CUDA/ROCm
- Moins flexible que raw PyTorch
""",
    alternatives_considered=[
        "HuggingFace Text Generation Inference (TGI)",
        "TensorRT-LLM (NVIDIA)",
        "Custom PyTorch implementation"
    ]
)


def print_adr(adr: ArchitectureDecisionRecord):
    """Affiche un ADR"""
    print(f"\n# {adr.id}: {adr.title}")
    print(f"Date: {adr.date}")
    print(f"Status: {adr.status.upper()}")
    print(f"\n## Context\n{adr.context}")
    print(f"\n## Decision\n{adr.decision}")
    print(f"\n## Consequences\n{adr.consequences}")


# ============================================================================
# 2. PERFORMANCE ET SCALING
# ============================================================================

"""
PERFORMANCE OPTIMIZATION CHECKLIST

Niveau 1: Modèle
  ✅ Quantization (GPTQ/AWQ) → 4x mémoire
  ✅ Flash Attention → 2-4x vitesse
  ✅ Model parallelism si >13B params
  ✅ Batch processing quand possible

Niveau 2: Infrastructure
  ✅ vLLM avec continuous batching
  ✅ GPU avec haute mémoire (A100 80GB)
  ✅ Fast storage (NVMe SSD)
  ✅ 10Gbps+ network

Niveau 3: Application
  ✅ Caching (Redis) pour requêtes fréquentes
  ✅ Request deduplication
  ✅ Async processing
  ✅ Connection pooling

Niveau 4: CDN/Edge
  ✅ CloudFlare pour static assets
  ✅ Edge functions pour routing
  ✅ Geographic distribution

Benchmarks Target:
  • Latency P50: <200ms
  • Latency P95: <500ms
  • Latency P99: <1000ms
  • Throughput: 1000+ tokens/sec
  • Availability: 99.9%
"""

class PerformanceMetrics:
    """Collecte et analyse métriques de performance"""

    def __init__(self):
        self.latencies = []
        self.throughputs = []

    def record_request(self, latency_ms: float, tokens: int):
        """Enregistre une requête"""
        self.latencies.append(latency_ms)
        self.throughputs.append(tokens / (latency_ms / 1000))

    def get_percentiles(self) -> Dict[str, float]:
        """Calcule percentiles de latence"""
        import numpy as np

        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)

        return {
            "p50": np.percentile(sorted_latencies, 50),
            "p95": np.percentile(sorted_latencies, 95),
            "p99": np.percentile(sorted_latencies, 99),
            "mean": np.mean(sorted_latencies),
            "max": max(sorted_latencies)
        }

    def meets_sla(self, p95_target: float = 500) -> bool:
        """Vérifie si SLA respecté"""
        percentiles = self.get_percentiles()
        return percentiles.get("p95", float('inf')) <= p95_target


# ============================================================================
# 3. SÉCURITÉ ET PRIVACY
# ============================================================================

"""
SECURITY CHECKLIST

Input Validation:
  ✅ Sanitize user inputs
  ✅ Length limits (prevent DoS)
  ✅ Rate limiting per user/IP
  ✅ Content filtering (toxic, PII)

Authentication & Authorization:
  ✅ API keys avec hash (SHA256)
  ✅ JWT tokens avec expiration
  ✅ OAuth2 pour enterprise
  ✅ RBAC (Role-Based Access Control)

Data Protection:
  ✅ Encryption at rest (AES-256)
  ✅ Encryption in transit (TLS 1.3)
  ✅ PII detection et masking
  ✅ Data retention policies

LLM-Specific:
  ✅ Prompt injection protection
  ✅ Output filtering (hallucinations, harmful content)
  ✅ Jailbreak detection
  ✅ Model watermarking

Compliance:
  ✅ GDPR (EU)
  ✅ CCPA (California)
  ✅ HIPAA (healthcare)
  ✅ SOC2 (enterprise)

Audit & Logging:
  ✅ Log all API calls
  ✅ User activity tracking
  ✅ Security events (failed auth, etc.)
  ✅ Regular security audits
"""

class SecurityValidator:
    """Validation sécurité des requêtes"""

    def __init__(self):
        # Patterns dangereux
        self.prompt_injection_patterns = [
            r"ignore previous instructions",
            r"you are now",
            r"system:",
            r"</s>",
            r"[INST]"
        ]

        # PII patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"
        }

    def validate_input(self, text: str) -> Dict[str, Any]:
        """
        Valide sécurité d'une entrée

        Returns:
            {
                "safe": bool,
                "issues": List[str],
                "pii_detected": Dict[str, List[str]]
            }
        """
        issues = []
        pii_found = {}

        # Vérifier prompt injection
        import re
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Potential prompt injection: {pattern}")

        # Détecter PII
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                pii_found[pii_type] = matches
                issues.append(f"PII detected: {pii_type}")

        # Vérifier longueur
        if len(text) > 10000:
            issues.append("Input too long (>10k chars)")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "pii_detected": pii_found
        }

    def mask_pii(self, text: str) -> str:
        """Masque les PII dans le texte"""
        import re

        masked = text

        # Masquer emails
        masked = re.sub(
            self.pii_patterns["email"],
            "[EMAIL_REDACTED]",
            masked
        )

        # Masquer téléphones
        masked = re.sub(
            self.pii_patterns["phone"],
            "[PHONE_REDACTED]",
            masked
        )

        return masked


# ============================================================================
# 4. COST OPTIMIZATION
# ============================================================================

"""
COST OPTIMIZATION STRATEGIES

1. Compute Optimization:
  ✅ Spot instances (-70% coût)
  ✅ Auto-scaling basé sur charge
  ✅ Scheduled scaling (nuit/weekend)
  ✅ GPU sharing entre modèles

2. Model Optimization:
  ✅ Quantization (4-bit) → 4x moins de GPU
  ✅ Distillation (small models pour tâches simples)
  ✅ Caching agressif
  ✅ Request batching

3. Data Transfer:
  ✅ Compression (gzip)
  ✅ CDN pour assets statiques
  ✅ Regional data centers

4. Licensing:
  ✅ Open-source models (LLaMA, Mistral)
  ✅ Reserved instances si usage stable
  ✅ Négocier contrats volume

Cost Breakdown Typique:
  • GPU compute: 60-70%
  • Storage (models + data): 10-15%
  • Network: 5-10%
  • Other services: 10-20%

Exemple Économies:
  Baseline: $40k/mois
    • EC2 p4d.24xlarge on-demand: $32.77/h × 720h = $23,594
    • Storage: $5,000
    • Network: $3,000
    • Services: $8,406

  Optimisé: $8k/mois (-80%)
    • Spot instances: $7,078 (-70%)
    • Quantization: Même perf avec instance plus petite
    • Caching: -50% requests
    • Résultat: $8,000/mois
"""

class CostTracker:
    """Tracking et optimisation des coûts"""

    def __init__(self):
        self.costs = []

    def log_request_cost(
        self,
        tokens: int,
        model: str,
        compute_time_seconds: float
    ) -> float:
        """
        Calcule coût d'une requête

        Formule:
          Cost = (GPU hours × hourly rate) + (tokens × token cost)
        """
        # Coûts par modèle (exemple)
        costs_per_model = {
            "llama-2-7b": {
                "gpu_hourly": 1.50,  # p3.2xlarge spot
                "per_1k_tokens": 0.0002
            },
            "llama-2-70b": {
                "gpu_hourly": 10.00,  # p4d.24xlarge spot
                "per_1k_tokens": 0.001
            }
        }

        model_costs = costs_per_model.get(model, costs_per_model["llama-2-7b"])

        # Coût compute
        gpu_cost = (compute_time_seconds / 3600) * model_costs["gpu_hourly"]

        # Coût tokens
        token_cost = (tokens / 1000) * model_costs["per_1k_tokens"]

        total = gpu_cost + token_cost

        self.costs.append({
            "timestamp": "now",
            "model": model,
            "tokens": tokens,
            "cost": total
        })

        return total

    def get_daily_cost(self) -> float:
        """Coût total journalier"""
        return sum(c["cost"] for c in self.costs)

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Répartition des coûts"""
        by_model = {}

        for cost_entry in self.costs:
            model = cost_entry["model"]
            by_model[model] = by_model.get(model, 0) + cost_entry["cost"]

        return by_model


# ============================================================================
# 5. MONITORING ET OBSERVABILITÉ
# ============================================================================

"""
MONITORING BEST PRACTICES

Golden Signals (Google SRE):
  1. Latency: Temps de réponse
  2. Traffic: Requêtes par seconde
  3. Errors: Taux d'erreur
  4. Saturation: Utilisation ressources

LLM-Specific Metrics:
  • Tokens per second
  • Cache hit rate
  • Model GPU utilization
  • Queue depth
  • Cost per request
  • TTFT (Time To First Token)

Alerts à Configurer:
  Critical:
    • Service down (>1 min)
    • Error rate >5%
    • P95 latency >2x normal
    • GPU OOM errors

  Warning:
    • Error rate >1%
    • P95 latency >1.5x normal
    • Cache hit rate <50%
    • Cost >20% budget

  Info:
    • New deployment
    • Traffic spike
    • Cache clear

Dashboards Essentiels:
  1. Overview (executives)
     • Requests/day
     • Availability %
     • Cost
     • User satisfaction

  2. Performance (engineers)
     • Latency percentiles
     • Throughput
     • Error rates
     • Resource utilization

  3. Business (product)
     • Active users
     • Top features
     • Retention
     • Revenue impact
"""

# Voir Chapitre 17 pour implémentation complète Prometheus + Grafana


# ============================================================================
# 6. TESTING ET QA
# ============================================================================

"""
TESTING STRATEGY

Niveau 1: Unit Tests
  ✅ Tokenization logic
  ✅ Prompt engineering functions
  ✅ Parsing outputs
  ✅ Validation logic

Niveau 2: Integration Tests
  ✅ LLM API calls
  ✅ Vector DB queries
  ✅ Cache behavior
  ✅ Authentication flow

Niveau 3: End-to-End Tests
  ✅ User workflows complets
  ✅ Multi-service orchestration
  ✅ Error recovery
  ✅ Performance sous charge

Niveau 4: LLM-Specific Tests
  ✅ Output quality (regression)
  ✅ Hallucination detection
  ✅ Bias testing
  ✅ Prompt injection resistance

Niveau 5: Load Testing
  ✅ Stress test (max capacity)
  ✅ Soak test (24h+ stable)
  ✅ Spike test (sudden traffic)
  ✅ Scalability test

Tools:
  • pytest: Unit/integration
  • locust: Load testing
  • k6: Performance testing
  • great_expectations: Data quality
"""

import pytest


class TestLLMQuality:
    """Tests qualité des outputs LLM"""

    def test_no_hallucination(self, llm_service):
        """Vérifie pas d'hallucination sur facts connus"""

        test_cases = [
            {
                "prompt": "Quelle est la capitale de la France?",
                "expected_answer": "Paris",
                "should_not_contain": ["Londres", "Berlin", "Madrid"]
            },
            {
                "prompt": "Combien font 2+2?",
                "expected_answer": "4",
                "should_not_contain": ["5", "3", "22"]
            }
        ]

        for case in test_cases:
            response = llm_service.complete(case["prompt"])

            # Vérifier réponse attendue présente
            assert case["expected_answer"].lower() in response.lower()

            # Vérifier pas de mauvaises réponses
            for wrong in case["should_not_contain"]:
                assert wrong.lower() not in response.lower()

    def test_output_length_control(self, llm_service):
        """Vérifie respect des limites de longueur"""

        max_tokens = 100
        response = llm_service.complete(
            "Écris un long texte",
            max_tokens=max_tokens
        )

        # Compter tokens (approximation)
        token_count = len(response.split()) * 1.3

        assert token_count <= max_tokens * 1.1  # 10% marge

    def test_consistent_outputs(self, llm_service):
        """Vérifie cohérence avec temperature=0"""

        prompt = "Liste 3 couleurs primaires"

        responses = [
            llm_service.complete(prompt, temperature=0)
            for _ in range(3)
        ]

        # Les 3 réponses devraient être identiques ou très similaires
        assert len(set(responses)) <= 2  # Au max 2 variantes


# ============================================================================
# 7. MLOps ET CI/CD
# ============================================================================

"""
MLOps PIPELINE

CI/CD pour LLMs:

  1. Code Changes:
     git push → GitHub Actions
     ├─ Linting (black, mypy)
     ├─ Unit tests
     ├─ Security scan
     └─ Build Docker image

  2. Model Changes:
     New model → Model Registry
     ├─ Validate format
     ├─ Run evals
     ├─ Compare to baseline
     └─ Tag version

  3. Deployment:
     Approved → Kubernetes
     ├─ Blue/Green deployment
     ├─ Canary (1% traffic)
     ├─ Monitor metrics
     └─ Promote or rollback

  4. Monitoring:
     Production → Dashboards
     ├─ Performance metrics
     ├─ Business metrics
     ├─ Cost tracking
     └─ Alerts if degradation

Model Registry Best Practices:
  ✅ Versioning (semantic)
  ✅ Metadata (training config, metrics)
  ✅ Lineage tracking
  ✅ Access control
  ✅ Promotion workflow (dev → staging → prod)

Deployment Strategies:
  • Blue/Green: 0 downtime, instant rollback
  • Canary: Gradual rollout (1% → 10% → 50% → 100%)
  • A/B Testing: Compare models
  • Shadow: New model in parallel (no user impact)
"""

# Exemple GitHub Actions workflow (YAML)
GITHUB_ACTIONS_WORKFLOW = """
name: LLM Service CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest black mypy

      - name: Lint
        run: |
          black --check .
          mypy .

      - name: Run tests
        run: pytest tests/ -v

      - name: Security scan
        run: bandit -r src/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t llm-service:${{ github.sha }} .

      - name: Push to registry
        run: docker push llm-service:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/llm-service \\
            llm-service=llm-service:${{ github.sha }}

          kubectl rollout status deployment/llm-service
"""


# ============================================================================
# 8. USER EXPERIENCE
# ============================================================================

"""
UX BEST PRACTICES pour LLM Apps

1. Gestion de la Latence:
  ✅ Streaming responses (token par token)
  ✅ Loading indicators clairs
  ✅ Optimistic UI updates
  ✅ Progress bars pour long tasks

2. Error Handling:
  ✅ Messages d'erreur clairs et actionnables
  ✅ Retry automatique (avec backoff)
  ✅ Fallbacks gracieux
  ✅ "Try again" button

3. Transparency:
  ✅ Montrer quand c'est l'IA qui répond
  ✅ Confidence scores
  ✅ Citations des sources (RAG)
  ✅ Limites du système explicites

4. Control:
  ✅ Permettre édition des réponses
  ✅ "Regenerate" option
  ✅ Feedback loop (👍 👎)
  ✅ Undo/Redo

5. Privacy:
  ✅ Expliquer utilisation des données
  ✅ Options de privacy
  ✅ Data deletion facile
  ✅ Pas de PII leak

Exemples de Bons Patterns:
  • ChatGPT: Streaming + regenerate + edit
  • Cursor: Inline suggestions + accept/reject
  • GitHub Copilot: Ghost text + tab to accept
  • Notion AI: Contextual + in-place editing
"""


# ============================================================================
# CHECKLIST PRODUCTION FINALE
# ============================================================================

def print_production_checklist():
    """Checklist avant mise en production"""
    print("\n" + "="*80)
    print("CHECKLIST PRODUCTION - LLM SYSTEM")
    print("="*80)

    checklist = {
        "Architecture": [
            "✓ Architecture documentée (ADRs)",
            "✓ Scaling strategy définie",
            "✓ Disaster recovery plan",
            "✓ Multi-region si critique"
        ],
        "Performance": [
            "✓ Load testing complété",
            "✓ Latency targets validés (P95 <500ms)",
            "✓ Auto-scaling configuré",
            "✓ Caching en place"
        ],
        "Security": [
            "✓ Security audit complété",
            "✓ Penetration testing fait",
            "✓ PII detection active",
            "✓ Encryption at rest et in transit",
            "✓ Rate limiting configuré",
            "✓ RBAC implémenté"
        ],
        "Monitoring": [
            "✓ Prometheus + Grafana opérationnels",
            "✓ Dashboards créés (overview, perf, business)",
            "✓ Alerts configurées (critical, warning)",
            "✓ Logging structuré avec ELK",
            "✓ Distributed tracing (Jaeger)",
            "✓ On-call rotation définie"
        ],
        "Cost": [
            "✓ Budget défini",
            "✓ Cost tracking en place",
            "✓ Alerts si dépassement",
            "✓ Optimization opportunities identifiées"
        ],
        "Testing": [
            "✓ Unit tests (>80% coverage)",
            "✓ Integration tests",
            "✓ E2E tests",
            "✓ LLM quality tests",
            "✓ Regression tests"
        ],
        "CI/CD": [
            "✓ Pipeline automatisé",
            "✓ Blue/Green deployment",
            "✓ Rollback plan testé",
            "✓ Canary deployment strategy"
        ],
        "Documentation": [
            "✓ API documentation (OpenAPI)",
            "✓ Runbooks pour incidents",
            "✓ Architecture diagrams",
            "✓ User guides",
            "✓ SDK documentation"
        ],
        "Compliance": [
            "✓ GDPR compliant (si EU)",
            "✓ SOC2 (si enterprise)",
            "✓ Terms of Service",
            "✓ Privacy Policy",
            "✓ Data retention policy"
        ],
        "User Experience": [
            "✓ Error messages clairs",
            "✓ Loading states",
            "✓ Feedback collection",
            "✓ A/B testing framework"
        ]
    }

    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")


# ============================================================================
# CONCLUSION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("BEST PRACTICES - SYNTHÈSE")
    print("="*80)

    # ADR Example
    print_adr(ADR_001)

    # Production Checklist
    print_production_checklist()

    print("\n\n" + "="*80)
    print("🎓 FÉLICITATIONS!")
    print("="*80)
    print("""
Vous avez complété "La Bible du Développeur AI/LLM 2026"!

Ce que vous maîtrisez maintenant:
  ✅ Fondations: Transformers, attention, tokenization
  ✅ Training: Fine-tuning, LoRA, RLHF, DPO
  ✅ Production: vLLM, quantization, deployment
  ✅ APIs: FastAPI, auth, rate limiting
  ✅ Observability: Prometheus, Grafana, tracing
  ✅ RAG: Vector DB, hybrid search, reranking
  ✅ Agents: ReAct, multi-agents, tool use
  ✅ Multimodal: Vision, audio, génération
  ✅ Long Context: RoPE, Flash Attention, 100k+ tokens
  ✅ Reasoning: CoT, Self-Consistency, ToT
  ✅ Projets: 15+ applications production-ready
  ✅ Best Practices: Architecture → Deployment → Monitoring

Vous êtes maintenant capable de:
  • Construire des systèmes LLM production
  • Fine-tuner des modèles pour cas spécifiques
  • Déployer à grande échelle (1000+ req/s)
  • Optimiser coûts (-80% possible)
  • Monitorer et maintenir en production
  • Innover avec techniques SOTA 2024-2025

Prochaines Étapes:
  1. Construire votre propre projet
  2. Contribuer à l'open-source (LLaMA, Mistral, etc.)
  3. Partager vos learnings avec la communauté
  4. Rester à jour (ce domaine évolue vite!)

Ressources pour Continuer:
  • Papers: arxiv.org/list/cs.CL/recent
  • Code: github.com/topics/llm
  • Community: Reddit r/LocalLLaMA, Discord servers
  • Courses: fast.ai, deeplearning.ai
  • Conférences: NeurIPS, ICML, ACL

Bonne chance dans votre carrière de développeur LLM! 🚀

---
Livre complet: ~85,000+ lignes de code production-ready
25 chapitres techniques + projets + best practices
Créé en 2024-2025 avec les techniques SOTA
    """)

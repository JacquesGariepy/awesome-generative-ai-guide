# Chapitre 17: Monitoring et Observabilité Avancée

## Introduction

L'**observabilité** est essentielle pour maintenir des LLMs en production. Il faut surveiller non seulement les métriques techniques, mais aussi la qualité des réponses.

### Les Trois Piliers de l'Observabilité

```python
"""
Observabilité = Comprendre l'état interne du système

Trois piliers:
  1. Métriques (Metrics):
     • Mesures numériques au fil du temps
     • Exemple: Latence, throughput, erreurs
     • Outil: Prometheus, Grafana

  2. Logs (Journaux):
     • Événements discrets
     • Exemple: Requêtes, erreurs, actions
     • Outil: ELK, Loki, Datadog

  3. Traces (Traces distribuées):
     • Flux de requêtes à travers services
     • Exemple: Requête API → Cache → vLLM → GPU
     • Outil: Jaeger, Zipkin, Tempo

Pour les LLMs, on ajoute:
  4. Qualité du modèle:
     • Cohérence des réponses
     • Toxicité, biais
     • Dérive du modèle (model drift)

Métriques importantes pour LLMs:
  • Performance technique:
    - Latence (p50, p95, p99)
    - Throughput (tokens/sec, requêtes/sec)
    - Taux d'erreur
    - Utilisation GPU/CPU/mémoire
    - Taille de la queue

  • Qualité du modèle:
    - Longueur des réponses
    - Taux de refus
    - Scores de toxicité
    - Similarité sémantique (vs attendu)
    - Feedback utilisateur

  • Business:
    - Coût par requête
    - Revenus par utilisateur
    - Taux de rétention
    - NPS (Net Promoter Score)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ModelMetrics:
    """Métriques pour monitoring du modèle"""
    timestamp: datetime

    # Métriques techniques
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    error_rate: float
    gpu_utilization: float
    gpu_memory_used_gb: float
    queue_length: int

    # Métriques qualité
    avg_response_length: float
    refusal_rate: float
    avg_toxicity_score: float
    cache_hit_rate: float

    # Métriques business
    cost_per_request: float
    active_users: int


def print_monitoring_overview():
    """Affiche un aperçu du monitoring"""
    print("="*80)
    print("MONITORING ET OBSERVABILITÉ POUR LLMs")
    print("="*80)

    print("""
DASHBOARD PRINCIPAL (Grafana):

┌─────────────────────────────────────────────────────────────────┐
│ Production LLM API - Vue d'ensemble                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Requêtes/sec: 150 ▲ 5%     Latence p95: 1.2s ▼ 10%           │
│ Taux d'erreur: 0.1% ▼      GPU Util: 85% ▲                    │
│                                                                 │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│ │ Latence      │  │ Throughput   │  │ Erreurs      │         │
│ │ p95: 1.2s    │  │ 150 req/s    │  │ 0.1%         │         │
│ │ ──────────── │  │ ──────────── │  │ ──────────── │         │
│ │   ╱╲    ╱╲   │  │     ╱─────   │  │   ──  ──     │         │
│ └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ Qualité du Modèle                                        │  │
│ │ • Longueur moy. réponse: 128 tokens                      │  │
│ │ • Taux de refus: 2.3%                                    │  │
│ │ • Score toxicité: 0.05 (faible)                          │  │
│ │ • Cache hit rate: 65%                                    │  │
│ └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ Ressources                                                │  │
│ │ GPU 1: [████████████────] 85%  12.5GB / 16GB            │  │
│ │ GPU 2: [████████████────] 82%  12.0GB / 16GB            │  │
│ │ CPU:   [████────────────] 35%                            │  │
│ │ RAM:   [██████──────────] 55%  44GB / 80GB              │  │
│ └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

ALERTES ACTIVES:
  ⚠️  Latence p95 élevée (> 2s) - Depuis 15 min
  ✅ Tous les autres indicateurs normaux

ACTIONS RECOMMANDÉES:
  • Vérifier la charge GPU
  • Considérer scaling horizontal (+2 instances)
  • Analyser les requêtes lentes (traces)
    """)


if __name__ == "__main__":
    print_monitoring_overview()
```

## 1. Prometheus et Grafana Avancés

```python
"""
Prometheus = Système de métriques time-series

Concepts:
  • Métrique: Mesure nommée (ex: latency_seconds)
  • Label: Dimension (ex: endpoint="/completions", status="200")
  • Scrape: Collecte périodique (toutes les 15s)
  • Query: PromQL pour analyser

Types de métriques:
  1. Counter: Toujours croissant (ex: total_requests)
  2. Gauge: Peut monter/descendre (ex: active_requests)
  3. Histogram: Distribution (ex: latency_seconds)
  4. Summary: Quantiles pré-calculés
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    Enum
)
from prometheus_client import make_asgi_app
from typing import Callable
import time


class AdvancedMetrics:
    """
    Métriques avancées pour LLM API

    Exemple:
        >>> metrics = AdvancedMetrics()
        >>> metrics.track_request(latency=1.5, tokens=128, model="llama-2-7b")
    """

    def __init__(self):
        # Métriques de base
        self.requests_total = Counter(
            'llm_requests_total',
            'Nombre total de requêtes',
            ['method', 'endpoint', 'status', 'model']
        )

        self.request_duration = Histogram(
            'llm_request_duration_seconds',
            'Durée des requêtes en secondes',
            ['endpoint', 'model'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )

        # Métriques spécifiques LLM
        self.tokens_generated = Counter(
            'llm_tokens_generated_total',
            'Nombre total de tokens générés',
            ['model', 'user_tier']
        )

        self.generation_duration = Histogram(
            'llm_generation_duration_seconds',
            'Durée de génération (sans overhead API)',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        self.tokens_per_second = Histogram(
            'llm_tokens_per_second',
            'Vitesse de génération (tokens/sec)',
            ['model'],
            buckets=[10, 20, 50, 100, 200, 500]
        )

        # Métriques de qualité
        self.response_length = Histogram(
            'llm_response_length_tokens',
            'Longueur des réponses en tokens',
            ['model'],
            buckets=[10, 50, 100, 200, 500, 1000, 2000]
        )

        self.toxicity_score = Histogram(
            'llm_toxicity_score',
            'Score de toxicité (0-1)',
            ['model'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.refusals_total = Counter(
            'llm_refusals_total',
            'Nombre de refus de répondre',
            ['model', 'reason']
        )

        # Métriques GPU
        self.gpu_utilization = Gauge(
            'llm_gpu_utilization_percent',
            'Utilisation GPU en pourcentage',
            ['gpu_id', 'node']
        )

        self.gpu_memory_used = Gauge(
            'llm_gpu_memory_used_bytes',
            'Mémoire GPU utilisée en bytes',
            ['gpu_id', 'node']
        )

        self.gpu_temperature = Gauge(
            'llm_gpu_temperature_celsius',
            'Température GPU en Celsius',
            ['gpu_id', 'node']
        )

        # Métriques cache
        self.cache_hits = Counter(
            'llm_cache_hits_total',
            'Nombre de hits cache',
            ['cache_type']
        )

        self.cache_misses = Counter(
            'llm_cache_misses_total',
            'Nombre de misses cache',
            ['cache_type']
        )

        # Métriques queue
        self.queue_length = Gauge(
            'llm_queue_length',
            'Nombre de requêtes en attente',
            ['priority']
        )

        self.queue_wait_time = Histogram(
            'llm_queue_wait_seconds',
            'Temps d\'attente dans la queue',
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
        )

        # Métriques business
        self.cost_total = Counter(
            'llm_cost_usd_total',
            'Coût total en USD',
            ['model', 'user_tier']
        )

        self.revenue_total = Counter(
            'llm_revenue_usd_total',
            'Revenus total en USD',
            ['user_tier']
        )

        # Info statique
        self.model_info = Info(
            'llm_model',
            'Informations sur le modèle chargé'
        )

    def track_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        latency: float,
        model: str,
        tokens_generated: int = 0,
        generation_time: float = 0.0,
        user_tier: str = "free",
        cache_hit: bool = False
    ):
        """Enregistre les métriques d'une requête"""
        # Requête
        self.requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status,
            model=model
        ).inc()

        self.request_duration.labels(
            endpoint=endpoint,
            model=model
        ).observe(latency)

        # Génération
        if tokens_generated > 0:
            self.tokens_generated.labels(
                model=model,
                user_tier=user_tier
            ).inc(tokens_generated)

            self.response_length.labels(model=model).observe(tokens_generated)

            if generation_time > 0:
                self.generation_duration.labels(model=model).observe(generation_time)

                tokens_per_sec = tokens_generated / generation_time
                self.tokens_per_second.labels(model=model).observe(tokens_per_sec)

        # Cache
        cache_type = "exact"  # ou "semantic"
        if cache_hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()


def demo_prometheus_queries():
    """Démo des requêtes Prometheus utiles"""
    print("\n" + "="*80)
    print("REQUÊTES PROMETHEUS (PromQL)")
    print("="*80)

    queries = {
        "Taux de requêtes (req/sec)": """
rate(llm_requests_total[5m])
        """,

        "Latence p95 par endpoint": """
histogram_quantile(0.95,
  rate(llm_request_duration_seconds_bucket[5m])
)
        """,

        "Taux d'erreur (%)": """
rate(llm_requests_total{status=~"5.."}[5m])
/
rate(llm_requests_total[5m]) * 100
        """,

        "Throughput moyen (tokens/sec)": """
rate(llm_tokens_generated_total[1m])
        """,

        "Cache hit rate (%)": """
rate(llm_cache_hits_total[5m])
/
(rate(llm_cache_hits_total[5m]) + rate(llm_cache_misses_total[5m]))
* 100
        """,

        "Utilisation GPU moyenne": """
avg(llm_gpu_utilization_percent)
        """,

        "Coût par requête (USD)": """
rate(llm_cost_usd_total[1h])
/
rate(llm_requests_total[1h])
        """,

        "Longueur moyenne des réponses": """
histogram_quantile(0.5,
  rate(llm_response_length_tokens_bucket[5m])
)
        """,

        "Requêtes lentes (> 5s) par heure": """
increase(
  llm_request_duration_seconds_bucket{le="5.0"}[1h]
)
        """,
    }

    print("\nREQUÊTES UTILES:\n")
    for name, query in queries.items():
        print(f"{name}:")
        print(query)
        print()


def demo_grafana_dashboards():
    """Démo de la configuration Grafana"""
    print("\n" + "="*80)
    print("DASHBOARDS GRAFANA")
    print("="*80)

    print("""
DASHBOARD 1: VUE D'ENSEMBLE TECHNIQUE

Rangée 1 - Métriques clés (Single Stat):
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Req/sec      │ Latence p95  │ Erreurs      │ GPU Util     │
  │ 150 ▲ 5%    │ 1.2s ▼ 10%  │ 0.1% ▼      │ 85% ▲       │
  └──────────────┴──────────────┴──────────────┴──────────────┘

Rangée 2 - Graphiques temps réel (Time Series):
  ┌────────────────────────────────────────────────────────────┐
  │ Latence par percentile                                     │
  │ ──── p50   ──── p95   ──── p99                            │
  │  3s │                              ╱─────p99               │
  │     │                      ╱──────╱                        │
  │  2s │              ╱──────╱                                │
  │     │      ╱──────╱─────p95                                │
  │  1s │ ────╱────p50                                         │
  │   0 └────────────────────────────────────────────────────  │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ Throughput (requêtes/sec)                                  │
  │ 200 │                                                       │
  │ 150 │     ╱────╲    ╱────╲                                 │
  │ 100 │ ───╱      ╲──╱      ╲───                            │
  │  50 │                                                       │
  │   0 └────────────────────────────────────────────────────  │
  └────────────────────────────────────────────────────────────┘

Rangée 3 - Ressources (Gauge + Heatmap):
  ┌──────────────┬──────────────┬──────────────────────────────┐
  │ GPU 1        │ GPU 2        │ Distribution latence         │
  │ [████░░] 85% │ [████░░] 82% │ 5s  [░░░░░░░░]              │
  │ 12.5GB/16GB  │ 12.0GB/16GB  │ 2s  [████████]              │
  │              │              │ 1s  [████████████]          │
  │              │              │ 0.5s[██████]                │
  └──────────────┴──────────────┴──────────────────────────────┘


DASHBOARD 2: QUALITÉ DU MODÈLE

Rangée 1 - Qualité (Stat + Gauge):
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Longueur moy.│ Taux refus   │ Toxicité     │ Cache hit    │
  │ 128 tokens   │ 2.3%         │ [░░] 0.05    │ 65%          │
  └──────────────┴──────────────┴──────────────┴──────────────┘

Rangée 2 - Distributions (Histogram):
  ┌────────────────────────────────────────────────────────────┐
  │ Distribution longueur réponses                             │
  │ 40% │     ████                                              │
  │ 30% │     ████  ████                                        │
  │ 20% │ ████████  ████  ██                                    │
  │ 10% │ ████████  ████  ██  ██  ██                           │
  │   0 └────────────────────────────────────────────────────  │
  │      10   50  100  200  500 1000 (tokens)                 │
  └────────────────────────────────────────────────────────────┘


DASHBOARD 3: BUSINESS METRICS

Rangée 1 - KPIs business:
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Coût/req     │ Revenue/user │ Users actifs │ Profit       │
  │ $0.002       │ $12.50       │ 1,234        │ $8,420       │
  └──────────────┴──────────────┴──────────────┴──────────────┘

Rangée 2 - Tendances:
  ┌────────────────────────────────────────────────────────────┐
  │ Coûts vs Revenus                                           │
  │      │ ──── Revenus   ──── Coûts   ──── Profit            │
  │ $20k │                                                      │
  │      │                              ╱────Revenue            │
  │ $15k │                      ╱──────╱                        │
  │      │              ╱──────╱                                │
  │ $10k │      ╱──────╱─────Profit                            │
  │      │ ────╱────Cost                                        │
  │  $5k │                                                      │
  └────────────────────────────────────────────────────────────┘


CONFIGURATION JSON (à importer):

{
  "dashboard": {
    "title": "LLM Production Monitoring",
    "panels": [
      {
        "title": "Requêtes/sec",
        "type": "stat",
        "targets": [{
          "expr": "rate(llm_requests_total[5m])"
        }]
      },
      {
        "title": "Latence p95",
        "type": "stat",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))"
        }]
      }
      // ... autres panels
    ]
  }
}

Disponible sur: https://grafana.com/grafana/dashboards/
Rechercher: "LLM monitoring" ou créer custom
    """)


if __name__ == "__main__":
    demo_prometheus_queries()
    demo_grafana_dashboards()
```

*[Suite avec Logs et Traces dans la partie 2...]*

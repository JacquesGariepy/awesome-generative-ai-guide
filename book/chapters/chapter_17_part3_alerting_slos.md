# Chapitre 17 (Partie 3): Alerting et SLOs

## 4. Système d'Alerting

```python
"""
Alerting = Notifier quand quelque chose ne va pas

Principes:
  • Alerter sur les symptômes, pas les causes
  • Éviter les fausses alertes (boy qui crie au loup)
  • Alertes actionnables (que faire?)
  • Plusieurs canaux (email, Slack, PagerDuty)

Types d'alertes:
  1. Critiques (P0):
     • Service down
     • Taux d'erreur élevé (> 5%)
     • Latence excessive (p95 > 10s)
     → Réveiller l'équipe on-call

  2. Importantes (P1):
     • Taux d'erreur modéré (> 1%)
     • Latence élevée (p95 > 5s)
     • Utilisation GPU > 95%
     → Notifier pendant heures de travail

  3. Avertissements (P2):
     • Tendances négatives
     • Capacité à 80%
     • Cache hit rate bas
     → Ticket pour investigation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import requests
import json


class AlertSeverity(Enum):
    """Niveau de sévérité d'alerte"""
    CRITIQUE = "critique"  # P0 - Réveiller on-call
    IMPORTANT = "important"  # P1 - Notifier rapidement
    AVERTISSEMENT = "avertissement"  # P2 - Ticket


@dataclass
class Alert:
    """Définition d'une alerte"""
    nom: str
    description: str
    severite: AlertSeverity
    condition: str  # Expression PromQL
    duree: str  # Durée avant déclenchement
    annotations: Dict[str, str]


# Alertes Prometheus
ALERTES_PRODUCTION = [
    Alert(
        nom="ServiceIndisponible",
        description="L'API LLM ne répond pas",
        severite=AlertSeverity.CRITIQUE,
        condition='up{job="llm-api"} == 0',
        duree="1m",
        annotations={
            "resume": "Service LLM indisponible",
            "description": "L'API ne répond pas depuis {{ $value }} minutes",
            "action": "1. Vérifier les pods Kubernetes\n2. Consulter les logs\n3. Redémarrer si nécessaire"
        }
    ),
    Alert(
        nom="TauxErreurEleve",
        description="Taux d'erreur supérieur à 5%",
        severite=AlertSeverity.CRITIQUE,
        condition="""
rate(llm_requests_total{status=~"5.."}[5m])
/
rate(llm_requests_total[5m]) > 0.05
        """,
        duree="5m",
        annotations={
            "resume": "Taux d'erreur élevé: {{ $value }}%",
            "description": "Plus de 5% des requêtes échouent",
            "action": "1. Vérifier les logs d'erreur\n2. Vérifier l'état du modèle\n3. Vérifier les ressources GPU"
        }
    ),
    Alert(
        nom="LatenceElevee",
        description="Latence p95 supérieure à 5 secondes",
        severite=AlertSeverity.IMPORTANT,
        condition="""
histogram_quantile(0.95,
  rate(llm_request_duration_seconds_bucket[5m])
) > 5
        """,
        duree="10m",
        annotations={
            "resume": "Latence p95 élevée: {{ $value }}s",
            "description": "95% des requêtes prennent plus de 5 secondes",
            "action": "1. Vérifier la charge\n2. Considérer scaling\n3. Analyser requêtes lentes avec tracing"
        }
    ),
    Alert(
        nom="UtilisationGPUElevee",
        description="GPU utilisé à plus de 95%",
        severite=AlertSeverity.IMPORTANT,
        condition='llm_gpu_utilization_percent > 95',
        duree="15m",
        annotations={
            "resume": "GPU {{ $labels.gpu_id }} à {{ $value }}%",
            "description": "Le GPU est saturé",
            "action": "1. Ajouter des GPUs\n2. Activer auto-scaling\n3. Vérifier si modèle trop gros"
        }
    ),
    Alert(
        nom="MemoireGPUElevee",
        description="Mémoire GPU presque pleine",
        severite=AlertSeverity.IMPORTANT,
        condition="""
llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes > 0.95
        """,
        duree="5m",
        annotations={
            "resume": "Mémoire GPU {{ $labels.gpu_id }} à {{ $value }}%",
            "description": "Risque de OOM (Out of Memory)",
            "action": "1. Réduire batch size\n2. Utiliser modèle quantisé\n3. Redémarrer pour libérer mémoire"
        }
    ),
    Alert(
        nom="CacheHitRateFaible",
        description="Taux de cache hit inférieur à 30%",
        severite=AlertSeverity.AVERTISSEMENT,
        condition="""
rate(llm_cache_hits_total[5m])
/
(rate(llm_cache_hits_total[5m]) + rate(llm_cache_misses_total[5m]))
< 0.30
        """,
        duree="30m",
        annotations={
            "resume": "Cache hit rate faible: {{ $value }}%",
            "description": "Opportunité d'optimisation",
            "action": "1. Analyser patterns de requêtes\n2. Ajuster stratégie de cache\n3. Augmenter TTL si approprié"
        }
    ),
]


def generer_config_prometheus_alertes():
    """Génère configuration Prometheus pour alertes"""
    print("="*80)
    print("CONFIGURATION PROMETHEUS ALERTMANAGER")
    print("="*80)

    print("""
# prometheus_rules.yml
groups:
  - name: llm_api_alerts
    interval: 30s
    rules:
""")

    for alerte in ALERTES_PRODUCTION:
        print(f"""
      - alert: {alerte.nom}
        expr: {alerte.condition}
        for: {alerte.duree}
        labels:
          severity: {alerte.severite.value}
          service: llm-api
        annotations:
          summary: "{alerte.annotations['resume']}"
          description: "{alerte.annotations['description']}"
          runbook: |
            {alerte.annotations['action']}
""")

    print("""
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    # Critiques → PagerDuty + Slack
    - match:
        severity: critique
      receiver: 'pagerduty-critique'
      continue: true

    - match:
        severity: critique
      receiver: 'slack-critique'

    # Importantes → Slack seulement
    - match:
        severity: important
      receiver: 'slack-important'

    # Avertissements → Email
    - match:
        severity: avertissement
      receiver: 'email-team'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#llm-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\\n{{ end }}'

  - name: 'pagerduty-critique'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .Annotations.summary }}'

  - name: 'slack-critique'
    slack_configs:
      - channel: '#llm-critical'
        color: 'danger'
        title: '🚨 CRITIQUE: {{ .GroupLabels.alertname }}'
        text: |
          *Résumé:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Actions:*
          ```
          {{ .Annotations.runbook }}
          ```

  - name: 'slack-important'
    slack_configs:
      - channel: '#llm-alerts'
        color: 'warning'
        title: '⚠️  IMPORTANT: {{ .GroupLabels.alertname }}'
        text: '{{ .Annotations.description }}'

  - name: 'email-team'
    email_configs:
      - to: 'team-llm@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alerts@example.com'
        auth_password: 'PASSWORD'
        headers:
          Subject: '{{ .GroupLabels.alertname }}'

inhibit_rules:
  # Si service down, ne pas alerter sur latence
  - source_match:
      alertname: 'ServiceIndisponible'
    target_match:
      alertname: 'LatenceElevee'
    equal: ['service']
    """)


class SlackNotifier:
    """
    Envoi de notifications Slack

    Exemple:
        >>> notifier = SlackNotifier(webhook_url="https://hooks.slack.com/...")
        >>> notifier.send_alert("Latence élevée détectée!")
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(
        self,
        titre: str,
        message: str,
        severite: AlertSeverity = AlertSeverity.IMPORTANT,
        champs: Optional[Dict[str, str]] = None
    ):
        """Envoie une alerte sur Slack"""
        # Couleur selon sévérité
        couleurs = {
            AlertSeverity.CRITIQUE: "danger",
            AlertSeverity.IMPORTANT: "warning",
            AlertSeverity.AVERTISSEMENT: "#439FE0"
        }

        # Icône selon sévérité
        icones = {
            AlertSeverity.CRITIQUE: "🚨",
            AlertSeverity.IMPORTANT: "⚠️",
            AlertSeverity.AVERTISSEMENT: "ℹ️"
        }

        # Construire message
        attachment = {
            "color": couleurs[severite],
            "title": f"{icones[severite]} {titre}",
            "text": message,
            "ts": int(time.time())
        }

        # Ajouter champs si fournis
        if champs:
            attachment["fields"] = [
                {"title": k, "value": v, "short": True}
                for k, v in champs.items()
            ]

        payload = {
            "username": "LLM Monitoring",
            "icon_emoji": ":robot_face:",
            "attachments": [attachment]
        }

        # Envoyer
        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        return response.status_code == 200


def demo_alerting():
    """Démo du système d'alerting"""
    print("\n" + "="*80)
    print("EXEMPLE D'ALERTE SLACK")
    print("="*80)

    print("""
Message Slack pour alerte critique:

┌─────────────────────────────────────────────────────────────┐
│ 🚨 CRITIQUE: Taux d'erreur élevé                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Plus de 5% des requêtes échouent depuis 5 minutes          │
│                                                             │
│ Taux actuel: 8.5%                                          │
│ Service: llm-api                                           │
│ Environnement: production                                  │
│                                                             │
│ Actions recommandées:                                       │
│ 1. Vérifier les logs d'erreur                             │
│ 2. Vérifier l'état du modèle                              │
│ 3. Vérifier les ressources GPU                            │
│                                                             │
│ [View in Grafana] [View in Prometheus] [Acknowledge]      │
└─────────────────────────────────────────────────────────────┘


WORKFLOW DE RÉPONSE:

1. Alerte déclenchée
   ↓
2. Notification envoyée (PagerDuty + Slack)
   ↓
3. Ingénieur on-call notifié
   ↓
4. Acknowledge alerte (stop notifications)
   ↓
5. Investigation:
   - Consulter dashboard Grafana
   - Analyser logs dans Kibana
   - Examiner traces dans Jaeger
   ↓
6. Mitigation:
   - Rollback si deployment récent
   - Scale si problème de capacité
   - Fix si bug identifié
   ↓
7. Résolution et post-mortem
   ↓
8. Alerte résolue automatiquement


BEST PRACTICES:

✅ Définir seuils réalistes (basés sur SLOs)
✅ Durée avant alerte (éviter fausses alertes)
✅ Runbooks clairs (que faire?)
✅ Escalade automatique si non résolu
✅ Post-mortem pour incidents majeurs

❌ Ne PAS alerter sur métriques non actionnables
❌ Ne PAS alerter trop souvent (fatigue d'alerte)
❌ Ne PAS avoir des alertes vagues
    """)


if __name__ == "__main__":
    generer_config_prometheus_alertes()
    demo_alerting()
```

## 5. SLIs, SLOs et SLAs

```python
"""
SLI, SLO, SLA = Mesurer et garantir la fiabilité

SLI (Service Level Indicator):
  • Métrique mesurée (ex: latence, disponibilité)
  • Quantitatif, mesurable
  • Exemple: "Latence p95 des requêtes"

SLO (Service Level Objective):
  • Objectif cible pour SLI
  • Interne à l'équipe
  • Exemple: "Latence p95 < 2s pour 99.9% des requêtes"

SLA (Service Level Agreement):
  • Contrat avec clients
  • Conséquences si non respecté (remboursement)
  • Exemple: "99.9% uptime ou crédit de 10%"

Règle: SLA < SLO < 100%
  • Marge d'erreur entre SLO et SLA
  • Permet d'agir avant violation SLA
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class SLIType(Enum):
    """Types de SLIs"""
    DISPONIBILITE = "disponibilite"
    LATENCE = "latence"
    THROUGHPUT = "throughput"
    QUALITE = "qualite"


@dataclass
class SLI:
    """Service Level Indicator"""
    nom: str
    type: SLIType
    description: str
    requete_promql: str
    unite: str


@dataclass
class SLO:
    """Service Level Objective"""
    nom: str
    sli: SLI
    objectif: float  # Valeur cible
    fenetre: str  # Période (ex: "30d")
    budget_erreur: float  # % d'erreur acceptable


# SLIs pour API LLM
SLIS_LLM = [
    SLI(
        nom="disponibilite_api",
        type=SLIType.DISPONIBILITE,
        description="Pourcentage de requêtes réussies",
        requete_promql="""
sum(rate(llm_requests_total{status!~"5.."}[5m]))
/
sum(rate(llm_requests_total[5m]))
        """,
        unite="%"
    ),
    SLI(
        nom="latence_p95",
        type=SLIType.LATENCE,
        description="95e percentile de latence",
        requete_promql="""
histogram_quantile(0.95,
  rate(llm_request_duration_seconds_bucket[5m])
)
        """,
        unite="s"
    ),
    SLI(
        nom="latence_p99",
        type=SLIType.LATENCE,
        description="99e percentile de latence",
        requete_promql="""
histogram_quantile(0.99,
  rate(llm_request_duration_seconds_bucket[5m])
)
        """,
        unite="s"
    ),
    SLI(
        nom="throughput",
        type=SLIType.THROUGHPUT,
        description="Requêtes par seconde",
        requete_promql='rate(llm_requests_total[5m])',
        unite="req/s"
    ),
]


# SLOs pour API LLM
SLOS_LLM = [
    SLO(
        nom="disponibilite_mensuelle",
        sli=SLIS_LLM[0],  # disponibilite_api
        objectif=0.999,  # 99.9%
        fenetre="30d",
        budget_erreur=0.001  # 0.1% = ~43 minutes/mois
    ),
    SLO(
        nom="latence_p95_acceptable",
        sli=SLIS_LLM[1],  # latence_p95
        objectif=2.0,  # 2 secondes
        fenetre="7d",
        budget_erreur=0.05  # 5% des requêtes peuvent dépasser
    ),
    SLO(
        nom="latence_p99_acceptable",
        sli=SLIS_LLM[2],  # latence_p99
        objectif=5.0,  # 5 secondes
        fenetre="7d",
        budget_erreur=0.01  # 1% peuvent dépasser
    ),
]


def calculer_budget_erreur(slo: SLO, duree_fenetre_jours: int) -> Dict:
    """
    Calcule le budget d'erreur restant

    Exemple:
        >>> slo = SLOS_LLM[0]  # 99.9% disponibilité
        >>> budget = calculer_budget_erreur(slo, 30)
        >>> print(budget['minutes_totales'])  # 43.2 minutes
    """
    # Convertir pourcentage en minutes
    minutes_par_mois = duree_fenetre_jours * 24 * 60

    if slo.sli.type == SLIType.DISPONIBILITE:
        # Budget = (1 - objectif) × durée
        budget_pourcent = (1 - slo.objectif)
        minutes_totales = budget_pourcent * minutes_par_mois

    else:
        # Pour latence/throughput, budget = % de requêtes qui peuvent échouer
        minutes_totales = slo.budget_erreur * minutes_par_mois

    return {
        "minutes_totales": minutes_totales,
        "heures_totales": minutes_totales / 60,
        "jours_totales": minutes_totales / (60 * 24),
        "pourcentage": slo.budget_erreur * 100 if slo.budget_erreur < 1 else (1 - slo.objectif) * 100
    }


def demo_slos():
    """Démo des SLOs"""
    print("\n" + "="*80)
    print("SLIs, SLOs ET BUDGETS D'ERREUR")
    print("="*80)

    print("\nSLOs DÉFINIS:\n")
    print(f"{'SLO':<30} {'Objectif':<15} {'Fenêtre':<10} {'Budget Erreur'}")
    print("-"*80)

    for slo in SLOS_LLM:
        if slo.sli.type == SLIType.DISPONIBILITE:
            objectif_str = f"{slo.objectif*100:.1f}%"
        else:
            objectif_str = f"< {slo.objectif}{slo.sli.unite}"

        budget_str = f"{slo.budget_erreur*100:.1f}%"

        print(f"{slo.nom:<30} {objectif_str:<15} {slo.fenetre:<10} {budget_str}")

    print("\n" + "="*80)
    print("BUDGETS D'ERREUR (sur 30 jours)")
    print("="*80 + "\n")

    for slo in SLOS_LLM:
        if "mensuelle" in slo.nom or "30d" in slo.fenetre:
            budget = calculer_budget_erreur(slo, 30)

            print(f"{slo.nom}:")
            print(f"  Objectif: {slo.objectif*100 if slo.objectif < 1 else slo.objectif}{slo.sli.unite}")
            print(f"  Budget total: {budget['minutes_totales']:.1f} minutes "
                  f"({budget['heures_totales']:.1f}h)")
            print(f"  Soit ~{budget['pourcentage']:.2f}% de downtime acceptable\n")

    print("="*80)
    print("EXEMPLE DE CALCUL:")
    print("="*80)
    print("""
SLO: 99.9% disponibilité sur 30 jours

Calcul:
  • 30 jours = 30 × 24 × 60 = 43,200 minutes
  • Budget erreur = (1 - 0.999) = 0.1%
  • Downtime acceptable = 43,200 × 0.001 = 43.2 minutes/mois

Si on a déjà eu:
  • 20 minutes de downtime → Reste 23.2 minutes (OK)
  • 40 minutes de downtime → Reste 3.2 minutes (ATTENTION!)
  • 50 minutes de downtime → Budget dépassé! (VIOLATION SLO)

Actions si budget faible:
  1. Geler les déploiements risqués
  2. Focus sur stabilité
  3. Postposer features non critiques
  4. Augmenter monitoring
    """)

    print("\n" + "="*80)
    print("DASHBOARD SLO (Grafana)")
    print("="*80)
    print("""
┌─────────────────────────────────────────────────────────────┐
│ SLO Dashboard - 30 derniers jours                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Disponibilité (SLO: 99.9%)                                 │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Actuel: 99.92% ✅                                       ││
│ │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.92%           ││
│ │ ────────────────────────────────── 99.90% (SLO)         ││
│ │                                                         ││
│ │ Budget erreur:                                          ││
│ │ Consommé: 15.2 min / 43.2 min (35%)                    ││
│ │ [████████░░░░░░░░░░░░░░] Bon                           ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ Latence p95 (SLO: < 2s)                                    │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Actuel: 1.65s ✅                                        ││
│ │ ━━━━━━━━━━━━━━━━━ 1.65s                               ││
│ │ ────────────────────── 2.00s (SLO)                     ││
│ │                                                         ││
│ │ Budget erreur:                                          ││
│ │ Violé: 2.3% / 5% (46%)                                 ││
│ │ [█████████░░░░░░░░░░░] Acceptable                      ││
│ └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘


REQUÊTES PROMETHEUS POUR SLO:

# Disponibilité actuelle (7 jours)
(
  sum(rate(llm_requests_total{status!~"5.."}[7d]))
  /
  sum(rate(llm_requests_total[7d]))
) * 100

# Budget erreur restant
(
  0.001 - (1 - (
    sum(rate(llm_requests_total{status!~"5.."}[30d]))
    /
    sum(rate(llm_requests_total[30d]))
  ))
)

# Pourcentage de requêtes violant SLO latence
(
  sum(rate(llm_request_duration_seconds_bucket{le="2.0"}[7d]))
  /
  sum(rate(llm_request_duration_seconds_count[7d]))
) * 100
    """)


if __name__ == "__main__":
    demo_slos()

    print("\n" + "="*80)
    print("✅ CHAPITRE 17 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant l'observabilité complète:

1. Métriques (Prometheus + Grafana):
   ✅ Counter, Gauge, Histogram, Summary
   ✅ Métriques techniques (latence, throughput, erreurs)
   ✅ Métriques LLM (tokens, qualité, GPU)
   ✅ Dashboards Grafana complets
   ✅ Requêtes PromQL avancées

2. Logs Structurés:
   ✅ Format JSON pour analyse
   ✅ Corrélation avec request_id
   ✅ Stack ELK (Elasticsearch, Logstash, Kibana)
   ✅ Recherche et agrégation

3. Tracing Distribué:
   ✅ OpenTelemetry pour instrumentation
   ✅ Spans et contexte de propagation
   ✅ Jaeger pour visualisation
   ✅ Identification des goulots

4. Alerting:
   ✅ Alertes critiques, importantes, avertissements
   ✅ Configuration Alertmanager
   ✅ Notifications multi-canaux (Slack, PagerDuty, email)
   ✅ Runbooks actionnables

5. SLIs/SLOs/SLAs:
   ✅ Définition des objectifs
   ✅ Budgets d'erreur
   ✅ Dashboards SLO
   ✅ Prise de décision basée sur SLOs

Observabilité production:
  • Visibilité complète sur le système
  • Détection rapide des problèmes
  • Investigation efficace
  • Amélioration continue

Impact réel:
  • MTTR (Mean Time To Recover) réduit de 80%
  • Incidents détectés proactivement
  • Coûts optimisés grâce aux insights
  • Confiance client maintenue

Next: Chapitres 19-22 → Applications Avancées (Agents, Multimodal, etc.)
    """)
```

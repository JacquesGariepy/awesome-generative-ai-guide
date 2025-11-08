# Chapitre 17 (Partie 2): Logs Structurés et Tracing Distribué

## 2. Logs Structurés

```python
"""
Logs Structurés = Logs au format JSON pour faciliter l'analyse

Pourquoi structurer les logs:
  ✅ Facile à parser (jq, grep, outils)
  ✅ Corrélation avec métriques et traces
  ✅ Recherche rapide (Elasticsearch)
  ✅ Agrégation et analyse

Format JSON vs texte:
  Texte: "2024-01-01 12:00:00 - requête complétée en 1.5s, user=abc"
  JSON:  {"timestamp": "2024-01-01T12:00:00Z", "event": "request_completed",
          "latency_ms": 1500, "user_id": "abc", "request_id": "xyz123"}

Avantages JSON:
  • Types de données (nombre, string, boolean)
  • Recherche par champ exact
  • Agrégations faciles
  • Compatibilité avec ELK stack
"""

import structlog
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class StructuredLogger:
    """
    Logger structuré pour LLM API

    Exemple:
        >>> logger = StructuredLogger(service_name="llm-api")
        >>> logger.info(
        ...     "requete_completee",
        ...     request_id="abc123",
        ...     latence_ms=1500,
        ...     tokens=128
        ... )
    """

    def __init__(
        self,
        service_name: str = "llm-api",
        environment: str = "production",
        log_level: str = "INFO"
    ):
        self.service_name = service_name
        self.environment = environment

        # Configuration structlog
        structlog.configure(
            processors=[
                # Ajouter contexte
                structlog.contextvars.merge_contextvars,
                # Ajouter timestamp
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                # Ajouter niveau de log
                structlog.stdlib.add_log_level,
                # Formater en JSON
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Créer logger
        self.logger = structlog.get_logger()

        # Ajouter contexte global
        self.logger = self.logger.bind(
            service=service_name,
            environment=environment
        )

    def info(self, event: str, **kwargs):
        """Log événement info"""
        self.logger.info(event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log événement erreur"""
        self.logger.error(event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log événement warning"""
        self.logger.warning(event, **kwargs)

    def debug(self, event: str, **kwargs):
        """Log événement debug"""
        self.logger.debug(event, **kwargs)

    def bind(self, **kwargs) -> 'StructuredLogger':
        """
        Créer logger avec contexte supplémentaire

        Exemple:
            >>> request_logger = logger.bind(request_id="abc123")
            >>> request_logger.info("debut_requete")
            >>> request_logger.info("fin_requete")
        """
        new_logger = StructuredLogger(self.service_name, self.environment)
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger


# Middleware de logging
class LoggingMiddleware:
    """
    Middleware FastAPI pour logging structuré

    Exemple:
        app.add_middleware(LoggingMiddleware, logger=logger)
    """

    def __init__(self, app, logger: StructuredLogger):
        self.app = app
        self.logger = logger

    async def __call__(self, request: Request, call_next):
        """Log chaque requête"""
        import time
        import uuid

        # Générer ID de requête
        request_id = str(uuid.uuid4())
        debut = time.time()

        # Créer logger avec contexte requête
        request_logger = self.logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host
        )

        # Log début requête
        request_logger.info(
            "requete_recue",
            user_agent=request.headers.get("user-agent")
        )

        try:
            # Traiter requête
            response = await call_next(request)

            # Calculer latence
            latence_ms = (time.time() - debut) * 1000

            # Log succès
            request_logger.info(
                "requete_completee",
                status_code=response.status_code,
                latence_ms=round(latence_ms, 2)
            )

            return response

        except Exception as e:
            # Calculer latence
            latence_ms = (time.time() - debut) * 1000

            # Log erreur
            request_logger.error(
                "requete_echouee",
                error_type=type(e).__name__,
                error_message=str(e),
                latence_ms=round(latence_ms, 2)
            )

            raise


def demo_logs_structures():
    """Démo des logs structurés"""
    print("="*80)
    print("LOGS STRUCTURÉS")
    print("="*80)

    print("""
EXEMPLE DE LOGS:

# Log événement simple
{"timestamp": "2024-01-01T12:00:00.123Z", "level": "info", "event": "requete_recue",
 "service": "llm-api", "environment": "production", "request_id": "abc123",
 "method": "POST", "path": "/v1/completions"}

# Log avec contexte métier
{"timestamp": "2024-01-01T12:00:01.456Z", "level": "info", "event": "generation_demarree",
 "service": "llm-api", "request_id": "abc123", "model": "llama-2-7b",
 "prompt_tokens": 42, "max_tokens": 256}

# Log de complétion
{"timestamp": "2024-01-01T12:00:02.789Z", "level": "info", "event": "requete_completee",
 "service": "llm-api", "request_id": "abc123", "status_code": 200,
 "latence_ms": 2666, "tokens_generes": 128, "tokens_par_sec": 48.0}

# Log d'erreur
{"timestamp": "2024-01-01T12:00:05.123Z", "level": "error", "event": "erreur_generation",
 "service": "llm-api", "request_id": "xyz789", "error_type": "OutOfMemoryError",
 "error_message": "CUDA out of memory", "gpu_id": 0, "gpu_memory_used_gb": 15.8}


RECHERCHE AVEC JQ:

# Filtrer par event
cat logs.jsonl | jq 'select(.event == "requete_completee")'

# Calculer latence moyenne
cat logs.jsonl | jq -s 'map(select(.event == "requete_completee") | .latence_ms) | add / length'

# Compter erreurs par type
cat logs.jsonl | jq -s 'group_by(.error_type) | map({type: .[0].error_type, count: length})'

# Requêtes lentes (> 5s)
cat logs.jsonl | jq 'select(.latence_ms > 5000)'


ELASTICSEARCH QUERIES:

# Recherche par ID de requête
GET /logs-*/_search
{
  "query": {
    "match": {
      "request_id": "abc123"
    }
  }
}

# Agrégation latence moyenne par endpoint
GET /logs-*/_search
{
  "size": 0,
  "aggs": {
    "par_endpoint": {
      "terms": {"field": "path"},
      "aggs": {
        "latence_moyenne": {
          "avg": {"field": "latence_ms"}
        }
      }
    }
  }
}

# Top 10 erreurs
GET /logs-*/_search
{
  "query": {"match": {"level": "error"}},
  "size": 0,
  "aggs": {
    "top_erreurs": {
      "terms": {
        "field": "error_type",
        "size": 10
      }
    }
  }
}


STACK ELK (Elasticsearch, Logstash, Kibana):

# docker-compose.yml
version: '3'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"

# logstash.conf
input {
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  # Ajouter geo-IP si besoin
  if [client_ip] {
    geoip {
      source => "client_ip"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-llm-api-%{+YYYY.MM.dd}"
  }
}

# Envoyer logs à Logstash
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5000))

log_entry = {
    "timestamp": "2024-01-01T12:00:00Z",
    "event": "requete_completee",
    "latence_ms": 1500
}

sock.send(json.dumps(log_entry).encode() + b'\n')
    """)


if __name__ == "__main__":
    demo_logs_structures()
```

## 3. Tracing Distribué

```python
"""
Tracing Distribué = Suivre une requête à travers plusieurs services

Pourquoi tracer:
  • Identifier les goulots d'étranglement
  • Comprendre les dépendances entre services
  • Déboguer les erreurs complexes
  • Optimiser les performances

Concepts:
  • Trace: Parcours complet d'une requête
  • Span: Une opération dans la trace
  • Context: Propagation ID entre services

Exemple de trace LLM:
  Requête API
    ├─ Authentification (5ms)
    ├─ Rate limiting (2ms)
    ├─ Cache lookup (3ms)
    └─ Génération
        ├─ Tokenization (10ms)
        ├─ GPU inference (2000ms) ← Goulot!
        └─ Détokenization (5ms)

OpenTelemetry = Standard pour observabilité
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode
import time
from typing import Optional, Dict


class DistributedTracer:
    """
    Configuration du tracing distribué

    Exemple:
        >>> tracer = DistributedTracer(service_name="llm-api")
        >>> tracer.setup()
    """

    def __init__(
        self,
        service_name: str = "llm-api",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831
    ):
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.tracer = None

    def setup(self):
        """Configure le tracing"""
        # Créer provider
        provider = TracerProvider()

        # Configurer exporteur Jaeger
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.jaeger_host,
            agent_port=self.jaeger_port,
        )

        # Ajouter processeur de spans
        provider.add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )

        # Définir comme provider global
        trace.set_tracer_provider(provider)

        # Créer tracer
        self.tracer = trace.get_tracer(self.service_name)

        return self.tracer

    def instrument_fastapi(self, app):
        """Instrumenter FastAPI automatiquement"""
        FastAPIInstrumentor.instrument_app(app)


# Utilisation manuelle des spans
def exemple_tracing_manuel():
    """Exemple d'utilisation manuelle du tracing"""
    print("\n" + "="*80)
    print("TRACING DISTRIBUÉ")
    print("="*80)

    print("""
CONFIGURATION:

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Setup
tracer_setup = DistributedTracer(service_name="llm-api")
tracer = tracer_setup.setup()

# Instrumenter FastAPI automatiquement
app = FastAPI()
tracer_setup.instrument_fastapi(app)


UTILISATION MANUELLE:

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    # Créer span parent
    with tracer.start_as_current_span("completion_endpoint") as span:
        # Ajouter attributs
        span.set_attribute("model", "llama-2-7b")
        span.set_attribute("max_tokens", request.max_tokens)

        # Authentification (span enfant)
        with tracer.start_as_current_span("authenticate"):
            user = await authenticate(request)
            span.set_attribute("user_id", user.id)

        # Cache lookup
        with tracer.start_as_current_span("cache_lookup") as cache_span:
            cached = await cache.get(request.prompt)
            cache_span.set_attribute("cache_hit", cached is not None)

            if cached:
                return cached

        # Génération (span complexe)
        with tracer.start_as_current_span("generation") as gen_span:
            # Tokenization
            with tracer.start_as_current_span("tokenize"):
                tokens = tokenizer.encode(request.prompt)
                gen_span.set_attribute("prompt_tokens", len(tokens))

            # Inference GPU
            with tracer.start_as_current_span("gpu_inference") as gpu_span:
                gpu_span.set_attribute("gpu_id", 0)

                try:
                    output = await model.generate(tokens)
                    gpu_span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    gpu_span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    gpu_span.record_exception(e)
                    raise

            # Détokenization
            with tracer.start_as_current_span("detokenize"):
                text = tokenizer.decode(output)
                gen_span.set_attribute("completion_tokens", len(output))

        # Fin
        span.set_status(Status(StatusCode.OK))

        return {"text": text}


VISUALISATION DANS JAEGER:

Service: llm-api
Trace: abc123-def456
Durée totale: 2.5s

completion_endpoint                    [████████████████████████] 2.5s
├─ authenticate                        [█] 5ms
├─ cache_lookup                        [█] 3ms (cache_hit=false)
└─ generation                          [███████████████████████] 2.4s
   ├─ tokenize                         [█] 10ms (prompt_tokens=42)
   ├─ gpu_inference                    [██████████████████████] 2.3s (gpu_id=0)
   └─ detokenize                       [█] 5ms (completion_tokens=128)

Insights:
  • 92% du temps = GPU inference
  • Cache miss = opportunité d'optimisation
  • Authentification rapide (5ms) = OK


PROPAGATION ENTRE SERVICES:

# Service A appelle Service B
import requests
from opentelemetry.propagate import inject

# Service A
with tracer.start_as_current_span("appel_service_b") as span:
    headers = {}

    # Injecter contexte dans headers
    inject(headers)

    # Appel HTTP avec headers
    response = requests.post(
        "http://service-b/process",
        headers=headers,
        json={"data": "..."}
    )

# Service B reçoit et continue la trace
from opentelemetry.propagate import extract

@app.post("/process")
async def process(request: Request):
    # Extraire contexte
    context = extract(request.headers)

    # Continuer la trace
    with tracer.start_as_current_span("traitement", context=context):
        # Le span sera lié au span parent du service A!
        result = await process_data(data)

    return result


DÉPLOIEMENT JAEGER:

# Docker
docker run -d \\
  --name jaeger \\
  -p 6831:6831/udp \\
  -p 16686:16686 \\
  jaegertracing/all-in-one:latest

# UI: http://localhost:16686

# Kubernetes
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/crds/jaegertracing.io_jaegers_crd.yaml
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/operator.yaml


REQUÊTES UTILES:

# Trouver traces lentes (> 5s)
service=llm-api duration>5s

# Trouver erreurs
service=llm-api error=true

# Trouver par opération
service=llm-api operation=gpu_inference

# Par attribut personnalisé
service=llm-api model=llama-2-7b

# Par plage de temps
service=llm-api minDuration=2s maxDuration=10s


INTÉGRATION AVEC GRAFANA:

# Ajouter Jaeger comme data source dans Grafana
# Créer dashboard avec:
  • Nombre de traces par service
  • Latence p95 par opération
  • Taux d'erreur
  • Dépendances entre services (graphe)


ALTERNATIVES À JAEGER:

Zipkin:
  • Plus simple
  • Moins de features
  • Bon pour débuter

Tempo (Grafana):
  • Intégration native Grafana
  • Stockage efficace
  • Open source

Datadog APM:
  • Commercial
  • Très complet
  • Coûteux mais puissant
    """)


if __name__ == "__main__":
    exemple_tracing_manuel()

    print("\n" + "="*80)
    print("BEST PRACTICES")
    print("="*80)
    print("""
LOGGING:

1. Utiliser logs structurés (JSON)
2. Inclure request_id partout
3. Logger les événements importants:
   ✅ Début/fin de requête
   ✅ Appels externes
   ✅ Erreurs avec stack trace
   ✅ Actions business (génération, cache hit, etc.)
4. Ne PAS logger:
   ❌ Données sensibles (mots de passe, tokens API)
   ❌ PII (données personnelles)
   ❌ Trop de détails en production
5. Niveaux appropriés:
   • DEBUG: Développement seulement
   • INFO: Flux normal
   • WARNING: Situations inhabituelles
   • ERROR: Erreurs qui nécessitent attention

TRACING:

1. Instrumenter automatiquement (FastAPI, requests, etc.)
2. Ajouter spans manuels pour opérations importantes
3. Ajouter attributs pertinents (user_id, model, tokens, etc.)
4. Propager contexte entre services
5. Échantillonnage en production (tracer 1-10% des requêtes)

CORRÉLATION:

• Utiliser même request_id dans:
  - Logs
  - Métriques (labels)
  - Traces (span IDs)

• Permet de:
  - Voir métriques → logs → traces
  - Déboguer problèmes rapidement
  - Comprendre incidents
    """)
```

*[Suite avec Alerting et SLOs dans la partie 3...]*

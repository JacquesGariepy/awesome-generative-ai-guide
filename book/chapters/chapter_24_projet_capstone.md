# Chapitre 24: Projet Capstone - Plateforme LLM Enterprise
## AICore Platform : Intégration Complète End-to-End

## 24.1 Vision du Projet Capstone

Après 23 chapitres couvrant théorie et pratique des LLMs, ce projet final répond à une question cruciale : **comment tout assembler pour créer une plateforme production-ready ?**

### 24.1.1 Le Problème : Fragmentation Technologique

**Situation typique dans une entreprise :**

```
┌─────────────────────────────────────────────────┐
│         PAYSAGE ACTUEL (FRAGMENTÉ)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  • API OpenAI → $50k/mois                       │
│  • API Anthropic → $30k/mois                    │
│  • LLM self-hosted → maintenance complexe       │
│  • Vector DB isolée → pas intégrée             │
│  • Scripts Python éparpillés → pas scalable    │
│  • Monitoring manuel → incidents fréquents     │
│                                                 │
│  RÉSULTAT: $200k/an + 3 ingénieurs full-time   │
└─────────────────────────────────────────────────┘
```

**Coûts cachés :**
- Latence API tierces : 500-2000ms (incontrôlable)
- Vendor lock-in : migration difficile
- Données sortent du périmètre (compliance GDPR/HIPAA)
- Pas d'optimisation cross-services

### 24.1.2 La Solution : Plateforme Unifiée

**AICore Platform - Architecture Cible :**

```
┌──────────────────────────────────────────────────────────┐
│                  UNIFIED CONTROL PLANE                    │
│      • Single API • Unified auth • Cost tracking          │
└────────────┬──────────────────────────────────┬──────────┘
             │                                  │
    ┌────────▼────────┐              ┌─────────▼──────────┐
    │  LLM SERVICES   │              │   DATA SERVICES    │
    │                 │              │                    │
    │ • Completion    │              │ • RAG Pipeline     │
    │ • Chat          │◄─────────────┤ • Vector DB        │
    │ • Fine-tuning   │              │ • Embeddings       │
    │ • Agents        │              │ • Reranking        │
    └────────┬────────┘              └─────────┬──────────┘
             │                                  │
    ┌────────▼──────────────────────────────────▼──────────┐
    │            INFRASTRUCTURE LAYER                       │
    │  vLLM • Kubernetes • GPU Pool • Redis • Postgres     │
    └──────────────────────────────────────────────────────┘
```

**Bénéfices :**

| Métrique | Avant (Fragmenté) | Après (Unifié) | Gain |
|----------|------------------|----------------|------|
| **Coût opérationnel** | $200k/an | $60k/an | -70% |
| **Latence P95** | 1500ms | 300ms | -80% |
| **Disponibilité** | 98.5% | 99.9% | +1.4% |
| **Time-to-deploy nouvelle feature** | 3 semaines | 2 jours | -90% |
| **Ingénieurs nécessaires** | 3 FTE | 0.5 FTE | -83% |

## 24.2 Spécifications du Système

### 24.2.1 Exigences Fonctionnelles

**Services Core (Must-Have) :**

1. **LLM Inference**
   - Completions (GPT-style)
   - Chat multi-turn
   - Streaming responses
   - Multiple models (7B → 70B)

2. **RAG (Retrieval-Augmented Generation)**
   - Document indexing
   - Hybrid search (dense + sparse)
   - Reranking
   - Citation tracking

3. **Agents Autonomes**
   - Tool use (API calls, calculateur, etc.)
   - Multi-step reasoning (ReAct pattern)
   - Memory management

4. **Fine-Tuning**
   - LoRA/QLoRA training
   - Data preparation assistée
   - Evaluation automatique
   - Model versioning

**Features Plateforme (Nice-to-Have) :**

5. **Multi-Tenant**
   - Isolation données par client
   - Quotas configurables
   - Billing par tenant

6. **Observability**
   - Métriques temps réel (Prometheus)
   - Dashboards (Grafana)
   - Distributed tracing (Jaeger)
   - Logs structurés (ELK)

### 24.2.2 Exigences Non-Fonctionnelles

**Performance :**

```python
# SLOs (Service Level Objectives)
SLOs = {
    "latency_p50": "< 200ms",
    "latency_p95": "< 500ms",
    "latency_p99": "< 1000ms",
    "throughput": "> 1000 req/s",
    "concurrent_requests": "> 500",
}

# SLAs (Service Level Agreements)
SLAs = {
    "uptime": "99.9%",  # ~8h downtime/an
    "error_rate": "< 0.1%",
    "data_loss": "0%",
}
```

**Scalabilité :**

| Charge | Users Concurrents | Req/s | Coût Infrastructure |
|--------|-------------------|-------|---------------------|
| Small | 100 | 50 | $500/mois (1× GPU) |
| Medium | 1,000 | 500 | $3k/mois (5× GPU) |
| Large | 10,000 | 5,000 | $20k/mois (30× GPU) |
| Enterprise | 100,000 | 50,000 | $150k/mois (200× GPU) |

**Sécurité :**

✅ **Authentification** : JWT + API keys
✅ **Autorisation** : RBAC (Role-Based Access Control)
✅ **Encryption** : TLS 1.3 in-transit, AES-256 at-rest
✅ **Rate Limiting** : Token bucket algorithm
✅ **Audit Logging** : Toutes actions tracées
✅ **Compliance** : SOC2, GDPR, HIPAA-ready

## 24.3 Architecture Système

### 24.3.1 Vue d'Ensemble

**4 Layers Architecturaux :**

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: API GATEWAY                                   │
│  ────────────────────────────────────────────────       │
│  • FastAPI (Python 3.11)                                │
│  • Auth middleware (JWT validation)                     │
│  • Rate limiting (Redis-backed)                         │
│  • Request validation (Pydantic)                        │
│  • Response caching (10min TTL)                         │
│                                                         │
│  Pourquoi FastAPI?                                      │
│    → 30% plus rapide que Flask                          │
│    → Validation automatique via types                   │
│    → Async native (crucial pour I/O-bound)              │
│    → OpenAPI auto-généré                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: SERVICE ORCHESTRATION                         │
│  ────────────────────────────────────────────────       │
│  • Service Registry (Consul)                            │
│  • Load Balancer (NGINX)                                │
│  • Circuit Breaker (Hystrix pattern)                    │
│  • Retry logic with exponential backoff                 │
│                                                         │
│  Pourquoi Service Mesh?                                 │
│    → Resilience : 1 service down ≠ platform down        │
│    → Observability : distributed tracing                │
│    → Security : mTLS entre services                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: BUSINESS SERVICES                             │
│  ────────────────────────────────────────────────       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   LLM    │  │   RAG    │  │  Agents  │              │
│  │ Service  │  │ Service  │  │ Service  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                         │
│  Chaque service:                                        │
│    • Stateless (horizontal scaling facile)              │
│    • Indépendamment déployable                          │
│    • Health checks (/health, /ready)                    │
│    • Metrics endpoint (/metrics)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: DATA & COMPUTE                                │
│  ────────────────────────────────────────────────       │
│  • vLLM (inference engine) → GPU pool                   │
│  • Qdrant (vector DB) → SSD storage                     │
│  • PostgreSQL (metadata) → replicated                   │
│  • Redis (cache + rate limit) → cluster mode            │
│  • S3 (model storage, logs) → multi-region              │
└─────────────────────────────────────────────────────────┘
```

### 24.3.2 Décisions d'Architecture Clés

**1. Pourquoi vLLM pour l'Inférence ?**

| Alternative | Latence | Throughput | Complexité | Choix |
|-------------|---------|-----------|------------|-------|
| **vLLM** | 50ms | 1000+ req/s | Moyenne | ✅ Choisi |
| TensorRT-LLM | 40ms | 1200 req/s | Élevée | ❌ Trop complexe |
| HuggingFace `transformers` | 200ms | 100 req/s | Faible | ❌ Trop lent |
| Triton | 60ms | 900 req/s | Élevée | ❌ Overhead |

**Raison du choix :** vLLM offre le meilleur compromis performance/simplicité avec PagedAttention (voir Chapitre 12).

**2. Pourquoi Qdrant pour Vector DB ?**

```python
# Benchmark (1M vectors, 768 dims, top-10 search)
vector_db_benchmarks = {
    "Qdrant": {
        "latency_p95": "15ms",
        "memory_usage": "2GB",
        "ease_of_use": 9/10,
        "cost": "$$$"
    },
    "Pinecone": {
        "latency_p95": "25ms",  # API latency + network
        "memory_usage": "N/A (managed)",
        "ease_of_use": 10/10,
        "cost": "$$$$$"  # 5× plus cher
    },
    "FAISS": {
        "latency_p95": "10ms",
        "memory_usage": "4GB",  # Pas optimisé
        "ease_of_use": 5/10,  # Besoin wrapper
        "cost": "$"
    },
    "Milvus": {
        "latency_p95": "20ms",
        "memory_usage": "3GB",
        "ease_of_use": 7/10,
        "cost": "$$"
    }
}
```

**Raison du choix :** Qdrant = performance proche de FAISS + facilité proche de Pinecone, self-hosted donc contrôle total.

**3. Architecture Multi-Tenant : Approches**

**Option A : DB par Tenant (Rejected)**

```
tenant_1 → postgres_1, qdrant_1
tenant_2 → postgres_2, qdrant_2
```

❌ **Problèmes :**
- Coût infrastructure × N tenants
- Pas de partage de cache
- Gestion complexe (backups, upgrades)

**Option B : Schema par Tenant (Rejected)**

```
postgres → schema_tenant1, schema_tenant2
```

❌ **Problèmes :**
- Migrations complexes
- Pas applicable à Qdrant

**Option C : Row-Level avec tenant_id (✅ Choisi)**

```sql
-- Toutes tables ont tenant_id
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,  -- Clé de partition
    content TEXT,
    created_at TIMESTAMP,
    INDEX idx_tenant (tenant_id)
);

-- Row-Level Security (PostgreSQL)
CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.current_tenant'));
```

✅ **Avantages :**
- Infrastructure partagée → économies d'échelle
- Queries automatiquement filtrées
- Migrations simples

### 24.3.3 Flow de Requête Complet

**Exemple : RAG Query avec Citation Tracking**

```
1. CLIENT REQUEST
   ────────────────
   POST /v1/rag/query
   Headers:
     Authorization: Bearer jwt_token_here
     X-Tenant-ID: tenant_abc
   Body:
     {
       "query": "Quels sont les bénéfices du fine-tuning LoRA ?",
       "include_citations": true
     }

2. API GATEWAY (FastAPI)
   ──────────────────────
   • Validate JWT → extract user_id
   • Check tenant_id matches JWT
   • Rate limit check (Redis):
       INCR rate:tenant_abc:2024-01-15-14:30
       → 45 requests dans cette minute
       → Limit: 1000/min
       → ✅ PASS

3. SERVICE ORCHESTRATION
   ──────────────────────
   • Route to RAG Service (load balancer)
   • Add trace_id for distributed tracing
   • Set timeout: 5s

4. RAG SERVICE
   ────────────
   a) Embedding Generation (50ms)
      query_vector = encoder.encode(query)
      → [0.123, -0.456, ...]  # 768 dims

   b) Vector Search (15ms)
      results = qdrant.search(
          collection="docs_tenant_abc",  # Isolé !
          vector=query_vector,
          top_k=10,
          filter={"tenant_id": "tenant_abc"}  # Double check
      )
      → 10 documents trouvés

   c) Reranking (30ms)
      # Cross-encoder pour précision
      reranked = reranker.rank(query, results)
      → Top 3 documents finaux

   d) LLM Generation (200ms)
      context = "\n\n".join([doc.content for doc in reranked[:3]])

      prompt = f"""Réponds à la question en te basant UNIQUEMENT sur le contexte.
      Cite tes sources avec [1], [2], etc.

      CONTEXTE:
      [1] {reranked[0].content}
      [2] {reranked[1].content}
      [3] {reranked[2].content}

      QUESTION: {query}

      RÉPONSE:"""

      response = llm.generate(prompt, max_tokens=300)

   e) Citation Extraction (10ms)
      # Parse [1], [2] dans response
      citations = extract_citations(response, reranked)

5. RESPONSE
   ─────────
   Total: 305ms (50+15+30+200+10)

   {
     "answer": "Le fine-tuning LoRA offre 3 bénéfices majeurs [1]:...",
     "citations": [
       {
         "id": 1,
         "source": "chapter_07_lora.md",
         "excerpt": "LoRA réduit les paramètres entraînables de 99%..."
       }
     ],
     "metadata": {
       "latency_ms": 305,
       "tokens_used": 1250,
       "cost_usd": 0.00125,
       "trace_id": "abc-123-def"
     }
   }

6. METRICS RECORDING
   ──────────────────
   Prometheus:
     rag_requests_total{tenant="tenant_abc", status="success"} +1
     rag_latency_seconds{tenant="tenant_abc"} 0.305
     rag_cost_usd{tenant="tenant_abc"} 0.00125

   Logs (JSON):
     {
       "timestamp": "2024-01-15T14:30:45Z",
       "service": "rag",
       "tenant_id": "tenant_abc",
       "trace_id": "abc-123-def",
       "latency_ms": 305,
       "status": "success"
     }
```

## 24.4 Implémentation : Points Critiques

### 24.4.1 Authentification et Sécurité

**Architecture Auth :**

```python
# 1. JWT Token Structure
jwt_payload = {
    "sub": "user_123",              # User ID
    "tenant_id": "tenant_abc",      # Tenant ID (clé isolation)
    "roles": ["admin", "api_user"], # RBAC
    "tier": "enterprise",           # Pour rate limits
    "exp": 1704067200,              # Expiration
    "iat": 1704060000               # Issued at
}

# 2. Validation Middleware (FastAPI)
async def validate_auth(request: Request):
    """
    Valide JWT et extrait contexte utilisateur
    """
    # Extract token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")

    token = auth_header.split(" ")[1]

    # Verify JWT signature + expiration
    try:
        payload = jwt.decode(
            token,
            key=PUBLIC_KEY,
            algorithms=["RS256"]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

    # Set context (utilisé par tous services downstream)
    request.state.user_id = payload["sub"]
    request.state.tenant_id = payload["tenant_id"]
    request.state.roles = payload["roles"]
    request.state.tier = payload["tier"]

    return payload
```

**Best Practice : Defense in Depth**

```python
# ❌ NE JAMAIS faire confiance au client
@app.post("/rag/query")
async def rag_query(
    request: RAGRequest,
    tenant_id_from_header: str = Header(...)  # ❌ Pas sûr !
):
    # Client peut envoyer n'importe quel tenant_id
    pass

# ✅ TOUJOURS extraire du JWT vérifié
@app.post("/rag/query")
async def rag_query(
    request: RAGRequest,
    auth: dict = Depends(validate_auth)
):
    tenant_id = auth["tenant_id"]  # ✅ Vérifié cryptographiquement
    # Query avec filter tenant_id
    results = await rag_service.query(
        query=request.query,
        tenant_id=tenant_id  # Isolation garantie
    )
```

### 24.4.2 Rate Limiting : Token Bucket Implementation

**Pourquoi Token Bucket ?**

| Algorithme | Pros | Cons | Use Case |
|------------|------|------|----------|
| **Token Bucket** | Permet bursts, simple | Besoin storage (Redis) | ✅ API publiques |
| Fixed Window | Très simple | Effet "double dipping" | Logs, non-critique |
| Sliding Window | Précis | Complexe, coûteux | Facturation exacte |
| Leaky Bucket | Smooth traffic | Pas de bursts | Rate limiting strict |

**Implémentation Redis :**

```python
import redis
import time

class TokenBucketRateLimiter:
    """
    Token Bucket avec Redis pour distribution

    Principe:
      • Chaque tenant a un "seau" de tokens
      • Tokens se remplissent à rate constant (ex: 100/min)
      • Requête consomme 1 token
      • Si seau vide → rate limit exceeded
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rate_limit(
        self,
        tenant_id: str,
        max_tokens: int = 1000,  # Capacité seau
        refill_rate: float = 100,  # Tokens/minute
    ) -> dict:
        """
        Vérifie si requête autorisée

        Returns:
            {
                "allowed": bool,
                "remaining": int,
                "reset_at": timestamp
            }
        """
        key = f"rate_limit:{tenant_id}"
        now = time.time()

        # Lua script (atomic operation)
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        -- Get current state
        local state = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(state[1]) or max_tokens
        local last_refill = tonumber(state[2]) or now

        -- Refill tokens based on elapsed time
        local elapsed = now - last_refill
        local refill_amount = elapsed * (refill_rate / 60)  -- per second
        tokens = math.min(max_tokens, tokens + refill_amount)

        -- Try consume 1 token
        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end

        -- Save state
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- Auto-cleanup

        return {allowed, tokens}
        """

        result = self.redis.eval(
            lua_script,
            1,  # num keys
            key,
            max_tokens,
            refill_rate,
            now
        )

        allowed, remaining = result

        return {
            "allowed": bool(allowed),
            "remaining": int(remaining),
            "reset_at": int(now + 60)  # Next minute
        }
```

**Métriques Rate Limiting :**

```python
# Dashboardre Grafana
rate_limit_metrics = {
    "total_requests": "Counter",
    "rate_limited_requests": "Counter",  # Combien bloquées ?
    "rate_limit_utilization": "Gauge",   # % capacity utilisée
    "avg_tokens_remaining": "Gauge"      # Headroom disponible
}

# Alerte si trop de rate limits
# → Peut indiquer besoin upgrade tier OU attaque
```

### 24.4.3 Cost Tracking par Tenant

**Pourquoi c'est critique ?**

Sans cost tracking précis :
- ❌ Impossible de facturer correctement
- ❌ Pas de détection d'abus (1 tenant consomme tout)
- ❌ Pas d'optimisation (quels tenants coûtent cher ?)

**Implémentation :**

```python
@dataclass
class UsageMetrics:
    """Métriques d'utilisation par tenant"""
    tenant_id: str
    timestamp: datetime

    # LLM
    llm_requests: int
    llm_tokens_prompt: int
    llm_tokens_completion: int
    llm_cost_usd: float

    # RAG
    rag_requests: int
    rag_documents_searched: int
    rag_embedding_tokens: int
    rag_cost_usd: float

    # Compute
    gpu_seconds: float  # Pour fine-tuning
    gpu_cost_usd: float

    # Storage
    storage_gb: float
    storage_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return (
            self.llm_cost_usd +
            self.rag_cost_usd +
            self.gpu_cost_usd +
            self.storage_cost_usd
        )


class CostTracker:
    """Track costs en temps réel"""

    def __init__(self, db: Database):
        self.db = db

        # Pricing (configurable par modèle)
        self.pricing = {
            "llama-2-7b": {
                "per_1k_tokens": 0.0002
            },
            "mistral-7b": {
                "per_1k_tokens": 0.0002
            },
            "gpu_h100": {
                "per_hour": 2.50  # On-demand pricing
            },
            "storage": {
                "per_gb_month": 0.10
            }
        }

    async def record_llm_usage(
        self,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Enregistre utilisation LLM"""

        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * self.pricing[model]["per_1k_tokens"]

        await self.db.execute("""
            INSERT INTO usage_metrics (
                tenant_id, timestamp, service,
                tokens, cost_usd
            ) VALUES ($1, NOW(), 'llm', $2, $3)
        """, tenant_id, total_tokens, cost)

        # Prometheus metric
        cost_counter.labels(
            tenant=tenant_id,
            service="llm",
            model=model
        ).inc(cost)

    async def get_month_to_date_cost(self, tenant_id: str) -> float:
        """Coût du mois en cours"""

        result = await self.db.fetchone("""
            SELECT SUM(cost_usd)
            FROM usage_metrics
            WHERE tenant_id = $1
              AND timestamp >= date_trunc('month', NOW())
        """, tenant_id)

        return result[0] or 0.0
```

**Dashboard Coûts (pour clients) :**

```
┌─────────────────────────────────────────────────┐
│  ACME CORP - Usage Dashboard (January 2024)     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Total Cost MTD: $1,247.32                      │
│  Projected EOM: $2,150                          │
│  Budget: $3,000 (✅ 72% used)                   │
│                                                 │
│  Breakdown:                                     │
│    LLM Inference ........... $892.15 (71%)      │
│    RAG Queries ............. $201.45 (16%)      │
│    Fine-Tuning ............. $123.50 (10%)      │
│    Storage ................. $ 30.22 (3%)       │
│                                                 │
│  Top Expensive Requests:                        │
│    1. Fine-tune job ft-abc123 ..... $89.50      │
│    2. RAG query batch .............. $12.30     │
│                                                 │
│  Optimization Suggestions:                      │
│    • Use llama-2-7b instead of 13b → -40% cost  │
│    • Enable caching → -25% redundant queries    │
└─────────────────────────────────────────────────┘
```

## 24.5 Déploiement Kubernetes

### 24.5.1 Architecture Kubernetes

**Namespace Strategy :**

```yaml
# Namespaces pour isolation
namespaces:
  - aicore-production   # Services production
  - aicore-staging      # Tests pré-prod
  - aicore-monitoring   # Prometheus, Grafana
  - aicore-data         # Postgres, Redis, Qdrant
```

**Déploiement Services :**

```yaml
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: aicore-production
spec:
  replicas: 5  # Horizontal scaling
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v1.2.3
    spec:
      containers:
      - name: fastapi
        image: aicore/api-gateway:v1.2.3
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: JWT_PUBLIC_KEY
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: jwt-public-key

        # Resource limits (crucial!)
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"     # 0.5 CPU
          limits:
            memory: "1Gi"
            cpu: "1000m"    # 1 CPU

        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
# Service (load balancer interne)
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
spec:
  selector:
    app: api-gateway
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer  # Expose externe
```

**LLM Service (avec GPUs) :**

```yaml
# llm-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 3  # 3 pods avec GPU
  template:
    spec:
      containers:
      - name: vllm
        image: aicore/vllm:v0.2.7
        command:
          - python
          - -m
          - vllm.entrypoints.api_server
          - --model=/models/llama-2-7b
          - --tensor-parallel-size=1

        resources:
          limits:
            nvidia.com/gpu: 1  # 1× GPU par pod

        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true

      # Node selector (nodes avec GPU)
      nodeSelector:
        accelerator: nvidia-a100

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

### 24.5.2 Auto-Scaling Strategy

**Horizontal Pod Autoscaler (HPA) :**

```yaml
# hpa-api-gateway.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway

  minReplicas: 3   # Minimum (haute dispo)
  maxReplicas: 20  # Maximum (coût control)

  metrics:
  # Scale basé sur CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Target 70% CPU

  # Scale basé sur memory
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # Scale basé sur custom metric (requests/s)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"  # 1000 req/s par pod

  # Behavior (éviter flapping)
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Attendre 5min avant scale down
      policies:
      - type: Percent
        value: 50  # Max 50% pods removed à la fois
        periodSeconds: 60

    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immédiat
      policies:
      - type: Percent
        value: 100  # Doubler les pods si besoin
        periodSeconds: 15
```

**Résultats Auto-Scaling :**

| Heure | Charge (req/s) | Pods Actifs | CPU Avg | Coût/heure |
|-------|---------------|-------------|---------|------------|
| 02:00 | 50 | 3 (min) | 20% | $3 |
| 09:00 | 500 | 5 | 65% | $5 |
| 12:00 | 2000 | 12 | 75% | $12 |
| 15:00 | 5000 | 20 (max) | 85% | $20 |
| 22:00 | 100 | 3 | 25% | $3 |

**Économies** : Auto-scaling = $120/jour vs static 20 pods = $480/jour → **-75% coût**

## 24.6 Monitoring et Observabilité

### 24.6.1 Les 3 Pilliers

**1. Metrics (Prometheus + Grafana)**

```python
# Métriques exposées par chaque service
from prometheus_client import Counter, Histogram, Gauge

# Compteurs
requests_total = Counter(
    'aicore_requests_total',
    'Total requests',
    ['service', 'tenant', 'status']
)

# Histogrammes (distribution latence)
request_latency = Histogram(
    'aicore_request_duration_seconds',
    'Request latency',
    ['service', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Gauges (valeurs instantanées)
active_requests = Gauge(
    'aicore_active_requests',
    'Currently processing requests',
    ['service']
)

gpu_utilization = Gauge(
    'aicore_gpu_utilization_percent',
    'GPU usage',
    ['gpu_id']
)
```

**Dashboard Grafana - Vue d'ensemble :**

```
┌──────────────────────────────────────────────────────┐
│  AICore Platform - Overview Dashboard                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  🟢 Status: Healthy    Uptime: 99.97%                │
│                                                      │
│  ┌────────────────┐  ┌────────────────┐             │
│  │ Requests/s     │  │ Latency P95    │             │
│  │                │  │                │             │
│  │   📈 1,247     │  │   ⚡ 285ms     │             │
│  │                │  │                │             │
│  └────────────────┘  └────────────────┘             │
│                                                      │
│  ┌───────────────────────────────────────┐          │
│  │ GPU Utilization (6 GPUs)              │          │
│  │ ████████████████░░ 85%                │          │
│  └───────────────────────────────────────┘          │
│                                                      │
│  Error Rate: 0.03%  ✅                               │
│  Cost Today: $127.45 (projected: $3,950/month)      │
│                                                      │
│  Top Tenants by Usage:                              │
│    1. acme_corp .......... 45% requests             │
│    2. startup_xyz ........ 23%                      │
│    3. enterprise_abc ..... 18%                      │
└──────────────────────────────────────────────────────┘
```

**2. Logs (Structured JSON)**

```python
import structlog
import logging

# Configuration structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Utilisation dans code
logger.info(
    "rag_query_processed",
    tenant_id="tenant_abc",
    query_id="q_12345",
    latency_ms=285,
    documents_found=8,
    llm_tokens=1247,
    cost_usd=0.00125
)

# Output JSON (parsable par ELK, Loki, etc.)
# {
#   "event": "rag_query_processed",
#   "level": "info",
#   "timestamp": "2024-01-15T14:30:45.123Z",
#   "logger": "aicore.rag",
#   "tenant_id": "tenant_abc",
#   "query_id": "q_12345",
#   "latency_ms": 285,
#   "documents_found": 8,
#   "llm_tokens": 1247,
#   "cost_usd": 0.00125
# }
```

**3. Traces (Jaeger Distributed Tracing)**

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
tracer_provider = TracerProvider()
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
tracer_provider.add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer(__name__)

# Utilisation
async def process_rag_query(query: str, tenant_id: str):
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("tenant_id", tenant_id)
        span.set_attribute("query_length", len(query))

        # Step 1: Embedding
        with tracer.start_as_current_span("embedding_generation"):
            vector = await generate_embedding(query)

        # Step 2: Vector search
        with tracer.start_as_current_span("vector_search"):
            docs = await vector_db.search(vector)

        # Step 3: LLM generation
        with tracer.start_as_current_span("llm_generation"):
            response = await llm.generate(query, docs)

        return response
```

**Trace View (Jaeger UI) :**

```
Request: POST /v1/rag/query [trace_id: abc-123-def] (Total: 305ms)
│
├─ rag_query (305ms)
│  │
│  ├─ embedding_generation (50ms)
│  │  └─ api_call_sentence_transformer (48ms)
│  │
│  ├─ vector_search (15ms)
│  │  └─ qdrant_query (12ms)
│  │
│  └─ llm_generation (235ms)
│     ├─ build_prompt (2ms)
│     ├─ vllm_inference (230ms)  ← Bottleneck !
│     └─ parse_response (3ms)
```

## 24.7 Résultats et ROI

### 24.7.1 Métriques de Production (6 mois)

**Performance :**

```python
production_metrics_6_months = {
    "availability": {
        "uptime_percent": 99.94,  # Target: 99.9%
        "incidents_major": 2,
        "incidents_minor": 12,
        "mttr_minutes": 15,  # Mean Time To Recovery
    },

    "performance": {
        "latency_p50_ms": 145,   # Target: <200ms
        "latency_p95_ms": 420,   # Target: <500ms
        "latency_p99_ms": 980,   # Target: <1000ms
        "throughput_rps": 1850,  # Target: >1000
    },

    "scale": {
        "tenants_active": 127,
        "total_requests": 45_000_000,
        "total_tokens_processed": 12_000_000_000,
        "peak_concurrent_requests": 743,
        "max_autoscale_pods": 18,  # vs 20 max → jamais saturé
    },

    "cost_efficiency": {
        "cost_per_1k_tokens": 0.00018,  # Target: <0.001 → ✅
        "gpu_utilization_avg": 0.78,    # 78% → bon
        "cache_hit_rate": 0.35,         # 35% requests cached
        "cost_savings_vs_openai": 0.72, # -72% moins cher
    },

    "reliability": {
        "error_rate": 0.027,  # 0.027% = 27 errors per 100k
        "rate_limit_hit_rate": 0.003,  # Très peu de clients limités
        "circuit_breaker_trips": 5,    # 5 incidents évités
    }
}
```

**ROI Business :**

| Métrique | Avant (APIs tierces) | Après (AICore) | Impact |
|----------|---------------------|----------------|--------|
| **Coût mensuel** | $16,500 | $4,200 | -75% ($147k/an savings) |
| **Latence P95** | 1,250ms | 420ms | -66% (meilleure UX) |
| **Control** | ❌ Vendor lock-in | ✅ Full control | Migration facile |
| **Compliance** | ⚠️ Data leaves EU | ✅ Self-hosted | GDPR compliant |
| **Customization** | ❌ Limited | ✅ Full | Fine-tuning sur data interne |

### 24.7.2 Leçons Apprises

**Ce qui a bien fonctionné ✅**

1. **vLLM pour inférence** : 3× plus rapide que transformers natif
2. **Auto-scaling** : -60% coût compute vs static sizing
3. **Caching intelligent** : 35% requests servies du cache
4. **Multi-tenant row-level** : Simple, efficace, économique
5. **Structured logging** : Debugging 10× plus rapide

**Ce qui a été difficile ❌**

1. **GPU cold start** : 30s pour charger modèle → pool pré-chauffé
2. **Cost attribution fine-grained** : Besoin instrumentation partout
3. **Version management models** : Rolling updates complexes
4. **Monitoring initial** : Pris 3 semaines setup complet
5. **Qdrant scaling** : Needed sharding à 10M+ vectors

**Si c'était à refaire 🔄**

1. **Commencer avec moins de features** : MVP = LLM + RAG uniquement
2. **Monitoring dès J1** : Pas après 2 mois
3. **Load testing plus tôt** : Découvrir bottlenecks avant prod
4. **Documentation API auto** : OpenAPI dès le début
5. **Plus de tests d'intégration** : E2E tests critiques

---

## 24.8 Prochaines Évolutions

**Roadmap Q1-Q2 2024 :**

```
Q1 (Jan-Mar):
  ✅ Multimodal support (images + text)
  ✅ Streaming responses (SSE)
  ✅ GraphQL API (en plus de REST)
  🔄 SDK JavaScript (en cours)

Q2 (Apr-Jun):
  📋 Fine-tuning UI (no-code)
  📋 A/B testing framework
  📋 Cost optimization suggestions (ML-powered)
  📋 Multi-region deployment (US + EU)

Q3 (Jul-Sep):
  📋 On-premise deployment option
  📋 Audit logs compliance (SOC2)
  📋 Advanced analytics dashboard
```

---

## Résumé du Chapitre 24

### Ce que vous avez appris :

✅ **Architecture Plateforme LLM Enterprise**
- 4 layers : API Gateway, Orchestration, Services, Data/Compute
- Décisions techniques : vLLM, Qdrant, PostgreSQL, Redis
- Trade-offs analysés pour chaque choix

✅ **Multi-Tenant Architecture**
- Row-level isolation avec tenant_id
- Rate limiting par tenant (Token Bucket)
- Cost tracking précis

✅ **Sécurité Production**
- JWT authentication
- RBAC authorization
- Defense in depth
- Rate limiting, audit logs

✅ **Déploiement Kubernetes**
- Deployments, Services, HPA
- Auto-scaling dynamique → -75% coût
- GPU scheduling

✅ **Observability Complète**
- Metrics (Prometheus)
- Logs structurés (JSON)
- Distributed tracing (Jaeger)

✅ **ROI Démontré**
- -75% coûts vs APIs tierces
- -66% latence P95
- 99.94% uptime

### Techniques du Livre Intégrées :

Ce projet capstone a intégré **toutes** les techniques des 23 chapitres précédents :

| Chapitres | Techniques Utilisées |
|-----------|---------------------|
| **1-6** | Architecture Transformer, tokenization, embeddings |
| **7-10** | Fine-tuning LoRA, instruction tuning, RLHF |
| **11** | Pipeline données (batch + streaming) |
| **12** | vLLM inference optimisée, PagedAttention |
| **13** | Quantization GPTQ/AWQ |
| **14** | Kubernetes deployment, cloud multi-region |
| **15** | FastAPI, auth JWT, rate limiting |
| **16** | Security best practices, audit logging |
| **17** | Prometheus, Grafana, distributed tracing |
| **18** | RAG complet avec hybrid search |
| **19** | Agents autonomes (ReAct pattern) |
| **21** | Long context handling |
| **22** | Chain-of-Thought reasoning |

### Code Architecture à Retenir :

```python
# 1. Multi-layer architecture
API Gateway → Service Mesh → Business Services → Infrastructure

# 2. Isolation multi-tenant
WHERE tenant_id = :current_tenant  # Toujours filtrer !

# 3. Rate limiting (Token Bucket Redis)
tokens = refill_tokens(elapsed) - 1
if tokens >= 0: allow() else: deny()

# 4. Cost tracking
cost = (tokens / 1000) * price_per_1k_tokens
record_usage(tenant_id, service, cost)

# 5. Auto-scaling based on metrics
if cpu > 70% or requests/pod > 1000: scale_up()
if cpu < 30% for 5min: scale_down()
```

---

**🎯 Exercice Final :**

Implémentez une version minimale d'AICore Platform avec :
1. API Gateway (FastAPI) + auth JWT
2. LLM Service (vLLM ou OpenAI API)
3. RAG Service (Qdrant + embeddings)
4. Metrics (Prometheus)
5. Déployez sur Kubernetes (Minikube local OK)

**Temps estimé** : 1-2 semaines pour MVP fonctionnel.

**Prochaine étape :** Chapitre 25 - Best Practices et Patterns pour systèmes LLM en production.

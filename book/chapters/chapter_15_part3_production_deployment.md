# Chapitre 15 (Partie 3): Production Deployment et Scaling

## 5. Caching et Performance

```python
"""
Caching = Improve performance and reduce costs

Why cache:
  • Reduce latency (ms instead of seconds)
  • Reduce costs (no GPU inference)
  • Reduce load on model
  • Improve user experience

What to cache:
  • Exact prompt matches
  • Semantic similarity matches
  • Common queries
  • Static responses

Cache strategies:
  1. Exact match:
     • Simple, fast
     • Cache key = prompt hash
     • Low hit rate

  2. Semantic match:
     • Fuzzy matching
     • Cache key = embedding
     • Higher hit rate
     • More complex

  3. LRU (Least Recently Used):
     • Evict old entries
     • Bounded memory
     • Good for hot data

Cache stores:
  • Redis: In-memory, fast, distributed
  • Memcached: Simple, fast
  • Application cache: Fastest but not distributed
"""

import hashlib
import redis.asyncio as redis
from typing import Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class ExactMatchCache:
    """
    Exact match caching with Redis

    Example:
        >>> cache = ExactMatchCache(redis_client)
        >>> await cache.set("prompt", "response")
        >>> response = await cache.get("prompt")
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl: int = 3600,  # 1 hour
        prefix: str = "cache:"
    ):
        self.redis = redis_client
        self.ttl = ttl
        self.prefix = prefix

    def _get_key(self, prompt: str) -> str:
        """Get cache key from prompt"""
        # Hash prompt for consistent key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{self.prefix}{prompt_hash}"

    async def get(self, prompt: str) -> Optional[str]:
        """
        Get cached response

        Args:
            prompt: Input prompt

        Returns:
            Cached response or None
        """
        key = self._get_key(prompt)
        cached = await self.redis.get(key)

        if cached:
            return cached.decode()

        return None

    async def set(
        self,
        prompt: str,
        response: str,
        ttl: Optional[int] = None
    ):
        """
        Cache response

        Args:
            prompt: Input prompt
            response: Generated response
            ttl: Time to live (seconds)
        """
        key = self._get_key(prompt)
        ttl = ttl or self.ttl

        await self.redis.set(key, response, ex=ttl)

    async def clear(self):
        """Clear all cache entries"""
        # Get all keys with prefix
        keys = []
        async for key in self.redis.scan_iter(match=f"{self.prefix}*"):
            keys.append(key)

        if keys:
            await self.redis.delete(*keys)


class SemanticCache:
    """
    Semantic similarity caching

    Uses embeddings to find similar prompts

    Example:
        >>> cache = SemanticCache(redis_client, model_name="all-MiniLM-L6-v2")
        >>> await cache.set("What is AI?", "AI is...")
        >>> response = await cache.get("What's artificial intelligence?")  # Similar!
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        ttl: int = 3600
    ):
        self.redis = redis_client
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.model.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def get(self, prompt: str) -> Optional[str]:
        """
        Get cached response by semantic similarity

        Args:
            prompt: Input prompt

        Returns:
            Cached response if similar prompt found
        """
        # Get prompt embedding
        prompt_emb = self._get_embedding(prompt)

        # Get all cached embeddings
        # (In production, use vector DB like Qdrant)
        keys = []
        async for key in self.redis.scan_iter(match="semantic_cache:*"):
            keys.append(key)

        best_match = None
        best_similarity = 0.0

        for key in keys:
            # Get cached embedding and response
            data = await self.redis.get(key)
            if not data:
                continue

            import json
            cached = json.loads(data.decode())

            # Compute similarity
            cached_emb = np.array(cached["embedding"])
            similarity = self._cosine_similarity(prompt_emb, cached_emb)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached["response"]

        # Return if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match

        return None

    async def set(self, prompt: str, response: str):
        """
        Cache response with embedding

        Args:
            prompt: Input prompt
            response: Generated response
        """
        # Get embedding
        embedding = self._get_embedding(prompt)

        # Create cache entry
        import json
        cache_entry = {
            "prompt": prompt,
            "response": response,
            "embedding": embedding.tolist()
        }

        # Store with prompt hash as key
        key = f"semantic_cache:{hashlib.sha256(prompt.encode()).hexdigest()}"
        await self.redis.set(
            key,
            json.dumps(cache_entry),
            ex=self.ttl
        )


# Caching middleware
class CacheMiddleware:
    """
    Caching middleware for FastAPI

    Example:
        app.add_middleware(CacheMiddleware, cache=cache)
    """

    def __init__(
        self,
        app,
        cache: ExactMatchCache,
        cache_enabled: bool = True
    ):
        self.app = app
        self.cache = cache
        self.cache_enabled = cache_enabled

    async def __call__(self, request: Request, call_next):
        """Check cache before processing request"""

        # Only cache POST /completions
        if not self.cache_enabled or request.method != "POST":
            return await call_next(request)

        if "/completions" not in request.url.path:
            return await call_next(request)

        # Get request body
        body = await request.body()
        import json
        data = json.loads(body)

        prompt = data.get("prompt", "")

        # Check cache
        cached_response = await self.cache.get(prompt)

        if cached_response:
            # Return cached response
            logger.info(f"Cache hit for prompt: {prompt[:50]}...")

            return JSONResponse({
                "id": str(uuid.uuid4()),
                "text": cached_response,
                "model": "cached",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "created_at": time.time(),
                "latency_ms": 0.0,
                "cached": True
            })

        # Process request
        response = await call_next(request)

        # Cache response (for future requests)
        # Note: This is simplified - in production, parse response properly
        # await self.cache.set(prompt, response_text)

        return response


def demo_caching():
    """Demo caching setup"""
    print("="*80)
    print("CACHING STRATEGIES")
    print("="*80)

    print("""
EXACT MATCH CACHE:

# Setup
import redis.asyncio as redis

redis_client = redis.Redis(host='localhost', port=6379)
cache = ExactMatchCache(redis_client, ttl=3600)

# Use in endpoint
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Check cache
    cached = await cache.get(request.prompt)
    if cached:
        return {"text": cached, "cached": True}

    # Generate
    response = await generate(request.prompt)

    # Cache for future
    await cache.set(request.prompt, response)

    return {"text": response, "cached": False}


SEMANTIC CACHE:

# Setup (requires sentence-transformers)
semantic_cache = SemanticCache(
    redis_client,
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.95
)

# Similar prompts get cached response
await semantic_cache.set("What is AI?", "AI is artificial intelligence...")

# This will hit cache (95%+ similar)
response = await semantic_cache.get("What's artificial intelligence?")


CACHE PERFORMANCE:

Without cache:
  • Latency: 1000ms (generation time)
  • Cost: $0.002 per request

With cache (50% hit rate):
  • Average latency: 500ms (50% × 2ms + 50% × 1000ms)
  • Average cost: $0.001 per request
  • 50% latency reduction, 50% cost reduction

With cache (80% hit rate):
  • Average latency: 202ms (80% × 2ms + 20% × 1000ms)
  • Average cost: $0.0004 per request
  • 80% latency reduction, 80% cost reduction


CACHE INVALIDATION:

Strategies:
  1. TTL (Time To Live):
     • Expire after X seconds
     • Simple, automatic
     • Good for: Dynamic content

  2. LRU (Least Recently Used):
     • Evict old entries when full
     • Good for: Limited memory

  3. Manual invalidation:
     • Clear on model update
     • Clear specific prompts
     • Good for: Control

# Clear cache on model update
@app.post("/admin/update_model")
async def update_model():
    await cache.clear()
    # Load new model
    return {"status": "updated"}


DISTRIBUTED CACHING:

# Redis cluster for high availability
from redis.cluster import RedisCluster

redis_cluster = RedisCluster(
    startup_nodes=[
        {"host": "redis-node-1", "port": 6379},
        {"host": "redis-node-2", "port": 6379},
        {"host": "redis-node-3", "port": 6379}
    ]
)

cache = ExactMatchCache(redis_cluster)

# Now cache is distributed across nodes
# - High availability
# - Better performance
# - Scales horizontally


MONITORING CACHE:

from prometheus_client import Counter, Histogram

cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')
cache_latency = Histogram('cache_latency_seconds', 'Cache lookup latency')

# Track hit rate
hit_rate = cache_hits / (cache_hits + cache_misses)

# Alert if hit rate drops
- alert: LowCacheHitRate
  expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.3
  for: 10m
  annotations:
    summary: "Cache hit rate below 30%"
    """)


if __name__ == "__main__":
    demo_caching()
```

## 6. Load Balancing et Scaling

```python
"""
Scaling = Handle increasing load

Horizontal Scaling:
  • Add more API instances
  • Distribute load
  • No single point of failure

Vertical Scaling:
  • Bigger instances
  • More CPU/RAM
  • Limited by hardware

Load Balancing:
  • Nginx: Open source
  • HAProxy: High performance
  • AWS ALB: Managed
  • GCP Load Balancer: Managed

Strategies:
  1. Round Robin: Even distribution
  2. Least Connections: Send to least busy
  3. IP Hash: Sticky sessions
  4. Weighted: More to powerful servers
"""


def demo_nginx_config():
    """Demo Nginx load balancer configuration"""
    print("\n" + "="*80)
    print("LOAD BALANCING WITH NGINX")
    print("="*80)

    nginx_config = """
# nginx.conf

upstream llm_api {
    # Load balancing strategy
    least_conn;  # Send to server with least connections

    # Backend servers
    server api-1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api-2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api-3:8000 weight=1 max_fails=3 fail_timeout=30s;

    # Keepalive connections
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL certificates
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Client body size (for large prompts)
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    # Compression
    gzip on;
    gzip_types application/json text/plain;
    gzip_min_length 1000;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    # Proxy to backend
    location / {
        proxy_pass http://llm_api;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Keepalive
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # Health check
        proxy_next_upstream error timeout http_502 http_503 http_504;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://llm_api/health;
    }

    # Metrics endpoint (internal only)
    location /metrics {
        allow 10.0.0.0/8;  # Internal network
        deny all;
        proxy_pass http://llm_api/metrics;
    }
}

# Health check configuration
# Check every 5 seconds, mark unhealthy after 3 failures
health_check interval=5s fails=3 passes=2;
    """

    print(nginx_config)

    print("""
DEPLOY WITH DOCKER COMPOSE:

# docker-compose.yml
version: '3.8'

services:
  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api-1
      - api-2
      - api-3

  # API instances
  api-1:
    build: .
    environment:
      - WORKER_ID=1
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  api-2:
    build: .
    environment:
      - WORKER_ID=2
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  api-3:
    build: .
    environment:
      - WORKER_ID=3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # Redis for caching and rate limiting
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml


START SERVICES:

# Build and start
docker-compose up -d --scale api=5

# Scale dynamically
docker-compose up -d --scale api=10

# Check status
docker-compose ps


AUTO-SCALING WITH KUBERNETES:

# See Chapter 14 for full Kubernetes setup
# HPA automatically scales based on metrics

kubectl autoscale deployment llm-api \\
    --cpu-percent=70 \\
    --min=3 \\
    --max=20
    """)


def demo_production_deployment():
    """Demo complete production deployment"""
    print("\n" + "="*80)
    print("COMPLETE PRODUCTION DEPLOYMENT")
    print("="*80)

    print("""
ARCHITECTURE:

Internet
  ↓
Cloudflare (CDN + DDoS protection)
  ↓
AWS ALB / GCP Load Balancer
  ↓
nginx (rate limiting, SSL termination)
  ↓
API Instances (FastAPI + vLLM)
  ↓
Redis (cache + rate limiting)
PostgreSQL (user data, usage)


DEPLOYMENT CHECKLIST:

Infrastructure:
  ✅ Load balancer configured
  ✅ SSL certificates (Let's Encrypt)
  ✅ Health checks enabled
  ✅ Auto-scaling configured
  ✅ Multi-region deployment

Application:
  ✅ Environment variables set
  ✅ Secrets in vault (not in code)
  ✅ Database migrations run
  ✅ Redis connected
  ✅ Model loaded successfully

Security:
  ✅ API authentication enabled
  ✅ Rate limiting configured
  ✅ CORS properly configured
  ✅ Input validation (Pydantic)
  ✅ SQL injection prevention
  ✅ XSS prevention

Monitoring:
  ✅ Prometheus metrics exported
  ✅ Grafana dashboards created
  ✅ Alerts configured
  ✅ Log aggregation (ELK/Datadog)
  ✅ Error tracking (Sentry)
  ✅ Uptime monitoring (Pingdom/UptimeRobot)

Performance:
  ✅ Caching enabled
  ✅ Connection pooling
  ✅ Async endpoints
  ✅ Load testing completed
  ✅ CDN for static assets

Reliability:
  ✅ Multiple availability zones
  ✅ Database backups automated
  ✅ Disaster recovery plan
  ✅ Rollback strategy
  ✅ Circuit breakers


DEPLOYMENT WORKFLOW:

1. Development:
   git checkout -b feature/new-endpoint
   # Develop and test locally
   pytest tests/

2. CI/CD Pipeline (GitHub Actions):
   name: Deploy API

   on:
     push:
       branches: [main]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Run tests
           run: |
             pip install -r requirements.txt
             pytest tests/

     build:
       needs: test
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Build Docker image
           run: docker build -t llm-api:${{ github.sha }} .
         - name: Push to registry
           run: docker push llm-api:${{ github.sha }}

     deploy:
       needs: build
       runs-on: ubuntu-latest
       steps:
         - name: Deploy to Kubernetes
           run: |
             kubectl set image deployment/llm-api \\
               llm-api=llm-api:${{ github.sha }}
             kubectl rollout status deployment/llm-api

3. Canary Deployment:
   # Deploy to 10% of traffic first
   kubectl set image deployment/llm-api-canary \\
     llm-api=llm-api:${{ github.sha }}

   # Monitor metrics for 30 minutes
   # If OK, deploy to 100%

4. Rollback (if needed):
   kubectl rollout undo deployment/llm-api


PERFORMANCE BENCHMARKING:

# Load testing with Locust
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def completion(self):
        self.client.post("/v1/completions", json={
            "prompt": "Write a poem",
            "max_tokens": 256
        })

# Run
locust -f loadtest.py --host=https://api.example.com


EXPECTED PERFORMANCE:

Small deployment (3x g5.xlarge):
  • Throughput: 10-20 req/sec
  • Latency p95: 2-5 seconds
  • Cost: ~$2500/month

Medium deployment (10x g5.2xlarge):
  • Throughput: 50-100 req/sec
  • Latency p95: 1-3 seconds
  • Cost: ~$9000/month

Large deployment (50x g5.2xlarge + cache):
  • Throughput: 500-1000 req/sec
  • Latency p95: 0.5-2 seconds
  • Cost: ~$40,000/month


COST OPTIMIZATION:

1. Use spot instances: 70% savings
2. Enable caching: 50% fewer requests
3. Use quantized models: 4x cheaper GPUs
4. Auto-scale: Only run what you need
5. Multi-region: Cheaper regions when possible

Example savings:
  • Baseline: $40,000/month
  • With spot: $12,000/month (70% off)
  • With cache (50% hit): $6,000/month
  • With auto-scale (50% avg): $3,000/month
  • Total savings: 92.5% ($3k vs $40k!)
    """)


if __name__ == "__main__":
    demo_nginx_config()
    demo_production_deployment()

    print("\n" + "="*80)
    print("✅ CHAPITRE 15 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant:

1. FastAPI Application:
   ✅ Async endpoints
   ✅ Pydantic validation
   ✅ Streaming responses
   ✅ OpenAPI docs

2. Authentication:
   ✅ API keys (SHA256 hashing)
   ✅ JWT tokens
   ✅ OAuth2 (mentioned)
   ✅ Protected endpoints

3. Rate Limiting:
   ✅ Token bucket algorithm
   ✅ Sliding window
   ✅ Redis-based (distributed)
   ✅ Per-user limits
   ✅ Cost-based limiting

4. Monitoring:
   ✅ Prometheus metrics
   ✅ Structured logging
   ✅ Distributed tracing
   ✅ Grafana dashboards
   ✅ Alerting

5. Caching:
   ✅ Exact match (Redis)
   ✅ Semantic similarity
   ✅ Hit rate optimization
   ✅ Cache invalidation

6. Production Deployment:
   ✅ Load balancing (nginx)
   ✅ Auto-scaling
   ✅ Docker Compose
   ✅ Kubernetes
   ✅ CI/CD pipeline
   ✅ Canary deployments

Real-world impact:
  • 10-1000 req/sec (with scaling)
  • < 100ms API overhead
  • 99.9% uptime
  • 90%+ cost optimization
  • Production-ready!

Next: Chapter 17 → Advanced Monitoring (APM, Tracing, SLOs)
    """)
```

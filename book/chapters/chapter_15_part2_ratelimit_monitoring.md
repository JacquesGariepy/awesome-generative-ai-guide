# Chapitre 15 (Partie 2): Rate Limiting et Monitoring

## 3. Rate Limiting

```python
"""
Rate Limiting = Prevent API abuse

Why rate limit:
  • Prevent abuse
  • Fair usage
  • Cost control
  • Protect infrastructure

Algorithms:
  1. Fixed Window:
     • Simple: X requests per minute
     • Problem: Burst at window boundary

  2. Sliding Window:
     • More accurate
     • Smooth distribution
     • Higher memory

  3. Token Bucket:
     • Industry standard
     • Allow bursts
     • Refill over time

  4. Leaky Bucket:
     • Smooth output rate
     • Queue requests
     • Drop if full

Implementation:
  • In-memory: Simple but doesn't scale
  • Redis: Distributed, scalable
  • API Gateway: Offload to infrastructure
"""

from fastapi import Request, HTTPException, status
from typing import Callable
import redis.asyncio as redis
from datetime import datetime, timedelta
import time


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with Redis

    Example:
        >>> limiter = TokenBucketRateLimiter(
        ...     redis_client=redis_client,
        ...     rate=100,  # 100 requests
        ...     per=60     # per 60 seconds
        ... )
        >>> await limiter.check_rate_limit(user_id="user123")
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        rate: int = 100,
        per: int = 60,
        burst: int = None
    ):
        """
        Args:
            redis_client: Redis client
            rate: Number of requests
            per: Time period in seconds
            burst: Burst capacity (defaults to rate)
        """
        self.redis = redis_client
        self.rate = rate
        self.per = per
        self.burst = burst or rate

        # Refill rate (tokens per second)
        self.refill_rate = rate / per

    async def check_rate_limit(
        self,
        key: str,
        cost: int = 1
    ) -> Dict[str, any]:
        """
        Check rate limit

        Args:
            key: User/API key identifier
            cost: Token cost (default 1)

        Returns:
            Rate limit info

        Raises:
            HTTPException: If rate limit exceeded
        """
        now = time.time()
        bucket_key = f"rate_limit:{key}"

        # Get current bucket state
        bucket = await self.redis.get(bucket_key)

        if bucket is None:
            # First request - initialize bucket
            tokens = self.burst - cost
            last_refill = now

            await self.redis.set(
                bucket_key,
                f"{tokens}:{last_refill}",
                ex=self.per * 2  # Expire after 2x period
            )

            return {
                "allowed": True,
                "tokens_remaining": tokens,
                "reset_at": now + self.per
            }

        # Parse bucket state
        tokens_str, last_refill_str = bucket.decode().split(":")
        tokens = float(tokens_str)
        last_refill = float(last_refill_str)

        # Refill tokens based on time elapsed
        elapsed = now - last_refill
        refill = elapsed * self.refill_rate
        tokens = min(self.burst, tokens + refill)

        # Check if enough tokens
        if tokens >= cost:
            # Allow request
            tokens -= cost
            last_refill = now

            await self.redis.set(
                bucket_key,
                f"{tokens}:{last_refill}",
                ex=self.per * 2
            )

            return {
                "allowed": True,
                "tokens_remaining": int(tokens),
                "reset_at": now + self.per
            }
        else:
            # Rate limit exceeded
            retry_after = (cost - tokens) / self.refill_rate

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "reset_at": now + self.per
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + self.per))
                }
            )


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter

    More accurate than fixed window, prevents boundary bursts

    Example:
        >>> limiter = SlidingWindowRateLimiter(redis_client, rate=100, window=60)
        >>> await limiter.check_rate_limit("user123")
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        rate: int = 100,
        window: int = 60
    ):
        self.redis = redis_client
        self.rate = rate
        self.window = window

    async def check_rate_limit(self, key: str) -> Dict:
        """Check rate limit with sliding window"""
        now = time.time()
        window_key = f"sliding_window:{key}"

        # Remove old entries (outside window)
        await self.redis.zremrangebyscore(
            window_key,
            0,
            now - self.window
        )

        # Count requests in current window
        count = await self.redis.zcard(window_key)

        if count >= self.rate:
            # Rate limit exceeded
            # Get oldest entry in window
            oldest = await self.redis.zrange(window_key, 0, 0, withscores=True)

            if oldest:
                reset_at = oldest[0][1] + self.window
            else:
                reset_at = now + self.window

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "reset_at": reset_at
                },
                headers={
                    "X-RateLimit-Limit": str(self.rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at))
                }
            )

        # Add current request
        await self.redis.zadd(window_key, {str(now): now})

        # Set expiry
        await self.redis.expire(window_key, self.window * 2)

        return {
            "allowed": True,
            "requests_remaining": self.rate - count - 1,
            "reset_at": now + self.window
        }


# Middleware for rate limiting
class RateLimitMiddleware:
    """
    Rate limit middleware

    Example:
        app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
    """

    def __init__(self, app, redis_client: redis.Redis):
        self.app = app
        self.limiter = TokenBucketRateLimiter(
            redis_client,
            rate=100,  # 100 requests per minute
            per=60
        )

    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting"""

        # Extract user ID (from auth, IP, etc.)
        user_id = self._get_user_id(request)

        # Check rate limit
        try:
            rate_info = await self.limiter.check_rate_limit(user_id)

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.limiter.rate)
            response.headers["X-RateLimit-Remaining"] = str(rate_info["tokens_remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(rate_info["reset_at"]))

            return response

        except HTTPException as e:
            # Return rate limit error
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail,
                headers=e.headers
            )

    def _get_user_id(self, request: Request) -> str:
        """Extract user ID from request"""
        # Try to get from auth
        if hasattr(request.state, "user"):
            return request.state.user["user_id"]

        # Fall back to IP address
        return request.client.host


# Demo rate limiting
def demo_rate_limiting():
    """Demo rate limiting setup"""
    print("="*80)
    print("RATE LIMITING")
    print("="*80)

    print("""
SETUP WITH REDIS:

# 1. Install Redis
docker run -d -p 6379:6379 redis:alpine

# 2. Install Python client
pip install redis

# 3. Create Redis client
import redis.asyncio as redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=False
)

# 4. Add rate limiting middleware
from fastapi import FastAPI

app = FastAPI()

# Add middleware
app.add_middleware(
    RateLimitMiddleware,
    redis_client=redis_client
)


RATE LIMIT TIERS:

Free Tier:
  • 100 requests/minute
  • 1000 requests/day
  • Burst: 120 requests

Pro Tier:
  • 1000 requests/minute
  • 50,000 requests/day
  • Burst: 1500 requests

Enterprise:
  • Custom limits
  • Dedicated resources
  • SLA guarantees


RATE LIMIT HEADERS:

Response headers:
  X-RateLimit-Limit: 100        # Total limit
  X-RateLimit-Remaining: 75     # Remaining requests
  X-RateLimit-Reset: 1699564800 # Reset timestamp

Error response (429):
  {
    "error": "Rate limit exceeded",
    "retry_after": 30,
    "reset_at": 1699564800
  }


PER-ENDPOINT RATE LIMITS:

# Different limits for different endpoints
@app.post("/v1/completions")
@limiter.limit("10/minute")  # Expensive endpoint
async def completions():
    pass

@app.get("/health")
@limiter.exempt  # No limit
async def health():
    pass


COST-BASED RATE LIMITING:

# Charge different costs based on request
async def dynamic_cost(request: Request) -> int:
    data = await request.json()

    # Cost = tokens to generate
    max_tokens = data.get("max_tokens", 256)

    # 1 token = 1 unit
    return max_tokens // 10

await limiter.check_rate_limit(
    key=user_id,
    cost=dynamic_cost(request)
)


MONITORING RATE LIMITS:

# Track rate limit hits
from prometheus_client import Counter

rate_limit_counter = Counter(
    'rate_limit_hits_total',
    'Total rate limit hits',
    ['user_id', 'endpoint']
)

# Increment on hit
rate_limit_counter.labels(
    user_id=user_id,
    endpoint=request.url.path
).inc()


BEST PRACTICES:

✅ Use Redis for distributed rate limiting
✅ Set reasonable limits per tier
✅ Include rate limit headers
✅ Provide clear error messages
✅ Allow burst for better UX
✅ Monitor rate limit hits
✅ Implement exponential backoff for clients
    """)


if __name__ == "__main__":
    demo_rate_limiting()
```

## 4. Monitoring et Metrics

```python
"""
Monitoring = Observability for production APIs

Three Pillars:
  1. Metrics: Numbers (Prometheus)
  2. Logs: Events (structured logging)
  3. Traces: Requests (OpenTelemetry)

Metrics to track:
  • Request rate (req/sec)
  • Latency (p50, p95, p99)
  • Error rate (%)
  • Token usage (input/output)
  • Model performance (throughput)
  • Resource usage (CPU, GPU, memory)
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from prometheus_client import CollectorRegistry
from fastapi.responses import Response
import structlog
from typing import Optional
import json


# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)

logger_struct = structlog.get_logger()


# Prometheus metrics
class Metrics:
    """
    Prometheus metrics for LLM API

    Example:
        >>> metrics = Metrics()
        >>> metrics.requests_total.labels(method="POST", endpoint="/completions").inc()
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Request metrics
        self.requests_total = Counter(
            'llm_api_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'llm_api_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        # Generation metrics
        self.tokens_generated = Counter(
            'llm_api_tokens_generated_total',
            'Total tokens generated',
            ['model'],
            registry=self.registry
        )

        self.generation_duration = Histogram(
            'llm_api_generation_duration_seconds',
            'Generation duration in seconds',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Model metrics
        self.active_requests = Gauge(
            'llm_api_active_requests',
            'Number of active requests',
            registry=self.registry
        )

        self.model_load_time = Gauge(
            'llm_api_model_load_time_seconds',
            'Model load time in seconds',
            ['model'],
            registry=self.registry
        )

        # Error metrics
        self.errors_total = Counter(
            'llm_api_errors_total',
            'Total number of errors',
            ['type', 'endpoint'],
            registry=self.registry
        )

        # Rate limit metrics
        self.rate_limit_hits = Counter(
            'llm_api_rate_limit_hits_total',
            'Total rate limit hits',
            ['user_id'],
            registry=self.registry
        )


# Global metrics instance
metrics = Metrics()


# Metrics middleware
class MetricsMiddleware:
    """
    Middleware to collect metrics

    Example:
        app.add_middleware(MetricsMiddleware, metrics=metrics)
    """

    def __init__(self, app, metrics: Metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, request: Request, call_next):
        """Collect metrics for each request"""

        # Increment active requests
        self.metrics.active_requests.inc()

        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Record metrics
            duration = time.time() - start_time

            self.metrics.requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()

            self.metrics.request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)

            return response

        except Exception as e:
            # Record error
            self.metrics.errors_total.labels(
                type=type(e).__name__,
                endpoint=request.url.path
            ).inc()

            raise

        finally:
            # Decrement active requests
            self.metrics.active_requests.dec()


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint

    Exposes metrics in Prometheus format
    """
    return Response(
        content=generate_latest(metrics.registry),
        media_type=CONTENT_TYPE_LATEST
    )


# Structured logging
class StructuredLogger:
    """
    Structured logging for better observability

    Example:
        >>> logger = StructuredLogger()
        >>> logger.info("request_completed", user_id="user123", latency_ms=150)
    """

    def __init__(self):
        self.logger = structlog.get_logger()

    def info(self, event: str, **kwargs):
        """Log info event"""
        self.logger.info(event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error event"""
        self.logger.error(event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning event"""
        self.logger.warning(event, **kwargs)


# Request logging
structured_logger = StructuredLogger()


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Log requests with structured logging"""

    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        response = await call_next(request)

        latency = (time.time() - start_time) * 1000

        structured_logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency,
            user_agent=request.headers.get("user-agent")
        )

        return response

    except Exception as e:
        latency = (time.time() - start_time) * 1000

        structured_logger.error(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=str(e),
            error_type=type(e).__name__,
            latency_ms=latency
        )

        raise


# Demo monitoring
def demo_monitoring():
    """Demo monitoring setup"""
    print("\n" + "="*80)
    print("MONITORING & OBSERVABILITY")
    print("="*80)

    print("""
PROMETHEUS SETUP:

# 1. Install Prometheus
docker run -d -p 9090:9090 \\
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \\
  prom/prometheus

# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

# 2. Access Prometheus UI
http://localhost:9090


GRAFANA SETUP:

# 1. Install Grafana
docker run -d -p 3000:3000 grafana/grafana

# 2. Access Grafana
http://localhost:3000
# Login: admin / admin

# 3. Add Prometheus data source
# Configuration → Data Sources → Add Prometheus
# URL: http://prometheus:9090

# 4. Import dashboard
# Dashboard → Import → Upload JSON


EXAMPLE QUERIES:

Request rate:
  rate(llm_api_requests_total[5m])

Latency p95:
  histogram_quantile(0.95, rate(llm_api_request_duration_seconds_bucket[5m]))

Error rate:
  rate(llm_api_errors_total[5m]) / rate(llm_api_requests_total[5m])

Tokens per second:
  rate(llm_api_tokens_generated_total[1m])


ALERTS:

# High error rate
- alert: HighErrorRate
  expr: rate(llm_api_errors_total[5m]) / rate(llm_api_requests_total[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate: {{ $value }}%"

# High latency
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(llm_api_request_duration_seconds_bucket[5m])) > 5
  for: 5m
  annotations:
    summary: "High p95 latency: {{ $value }}s"


STRUCTURED LOGGING:

# Output format (JSON)
{
  "event": "request_completed",
  "request_id": "abc123",
  "method": "POST",
  "path": "/v1/completions",
  "status_code": 200,
  "latency_ms": 1250.5,
  "timestamp": "2024-01-01T12:00:00Z"
}

# Benefits:
✅ Easy to parse
✅ Can query with tools (jq, grep)
✅ Ship to log aggregators (ELK, Datadog)
✅ Correlate with traces


DISTRIBUTED TRACING (OpenTelemetry):

# 1. Install
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi

# 2. Setup
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# 3. View traces in Jaeger
http://localhost:16686


COMPLETE OBSERVABILITY STACK:

Metrics:
  • Prometheus: Collect metrics
  • Grafana: Visualize dashboards
  • Alertmanager: Send alerts

Logs:
  • Structured logging (JSON)
  • Fluentd/Logstash: Collect logs
  • Elasticsearch: Store logs
  • Kibana: Search/visualize logs

Traces:
  • OpenTelemetry: Instrument code
  • Jaeger/Zipkin: Store traces
  • Visualize request flow

APM (All-in-one):
  • Datadog: Commercial, best UX
  • New Relic: Commercial
  • Elastic APM: Open source
    """)


if __name__ == "__main__":
    demo_monitoring()
```

*[Suite avec Production Deployment dans la partie 3...]*

# Chapitre 15: APIs et Services pour LLMs

## Introduction

Les **APIs de production** doivent gérer l'authentication, le rate limiting, le monitoring, et scaler efficacement.

### Architecture d'une API LLM

```python
"""
Production LLM API = FastAPI + Auth + Rate Limiting + Monitoring + Scaling

Architecture:
  Client
    ↓
  Load Balancer (nginx/ALB)
    ↓
  API Gateway (rate limiting, auth)
    ↓
  FastAPI Service (async)
    ↓
  vLLM Backend
    ↓
  Model

Components:
  1. FastAPI:
     • Async Python framework
     • Automatic OpenAPI docs
     • Fast (comparable to Node.js/Go)
     • Type validation (Pydantic)

  2. Authentication:
     • API keys (simple)
     • JWT tokens (standard)
     • OAuth2 (enterprise)

  3. Rate Limiting:
     • Per-user limits
     • Token bucket algorithm
     • Redis-based (distributed)

  4. Monitoring:
     • Prometheus metrics
     • Structured logging
     • Distributed tracing (OpenTelemetry)

  5. Scaling:
     • Horizontal: Multiple API instances
     • Vertical: Better hardware
     • Caching: Redis for responses
     • Queue: Celery for async jobs

Performance targets:
  • Latency: < 100ms (API overhead)
  • Throughput: 1000+ req/sec
  • Availability: 99.9%+
  • Error rate: < 0.1%
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import asyncio
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    """Completion request schema"""
    prompt: str = Field(..., min_length=1, max_length=4096, description="Input prompt")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Stream response")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a poem about AI",
                "max_tokens": 256,
                "temperature": 0.8,
                "top_p": 0.95,
                "stream": False
            }
        }


class CompletionResponse(BaseModel):
    """Completion response schema"""
    id: str = Field(..., description="Unique request ID")
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage")
    created_at: float = Field(..., description="Timestamp")
    latency_ms: float = Field(..., description="Generation latency")

    class Config:
        schema_extra = {
            "example": {
                "id": "req_abc123",
                "text": "In circuits deep and logic bright...",
                "model": "llama-2-7b",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 128,
                    "total_tokens": 133
                },
                "created_at": 1699564800.0,
                "latency_ms": 1250.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    average_latency_ms: float


# Demo API structure
def demo_api_structure():
    """Demo API structure and components"""
    print("="*80)
    print("PRODUCTION LLM API STRUCTURE")
    print("="*80)

    print("""
DIRECTORY STRUCTURE:

llm-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── models.py            # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── completions.py   # Completion endpoints
│   │   ├── chat.py          # Chat endpoints
│   │   └── embeddings.py    # Embedding endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py           # LLM service (vLLM)
│   │   ├── cache.py         # Redis cache
│   │   └── queue.py         # Task queue
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── api_key.py       # API key auth
│   │   └── jwt.py           # JWT auth
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limit.py    # Rate limiting
│   │   ├── logging.py       # Request logging
│   │   └── metrics.py       # Prometheus metrics
│   └── config.py            # Configuration
├── tests/
│   ├── test_api.py
│   ├── test_auth.py
│   └── test_rate_limit.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── requirements.txt
├── .env.example
└── README.md


KEY COMPONENTS:

1. FastAPI App (main.py):
   • Define routes
   • Add middleware
   • Configure CORS
   • OpenAPI docs

2. Services:
   • LLM service: Interface to vLLM
   • Cache service: Redis for responses
   • Queue service: Celery for async tasks

3. Authentication:
   • API key validation
   • JWT token verification
   • User management

4. Middleware:
   • Rate limiting (Redis)
   • Request logging
   • Metrics collection (Prometheus)
   • Error handling

5. Configuration:
   • Environment variables
   • Model settings
   • Rate limits
   • Cache settings


TECH STACK:

Backend:
  • FastAPI: Web framework
  • Pydantic: Data validation
  • vLLM: LLM inference
  • Redis: Cache + rate limiting
  • PostgreSQL: User/usage data
  • Celery: Task queue

Monitoring:
  • Prometheus: Metrics
  • Grafana: Dashboards
  • Sentry: Error tracking
  • Datadog: APM (optional)

Deployment:
  • Docker: Containerization
  • Kubernetes: Orchestration
  • nginx: Reverse proxy
  • Let's Encrypt: SSL certificates
    """)


if __name__ == "__main__":
    demo_api_structure()
```

## 1. FastAPI Application

```python
"""
FastAPI = Modern Python web framework

Features:
  ✅ Fast: Comparable to Node.js/Go
  ✅ Async: Native async/await support
  ✅ Type hints: Pydantic validation
  ✅ Auto docs: Swagger UI + ReDoc
  ✅ Easy testing: Built-in test client
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import uuid
from typing import AsyncGenerator
import json


# Global state
class AppState:
    """Application state"""
    def __init__(self):
        self.llm = None
        self.start_time = time.time()
        self.total_requests = 0
        self.total_latency = 0.0

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    # Startup
    logger.info("Starting LLM API...")

    # Load model
    logger.info("Loading model...")
    try:
        # Simulate loading (in production, load vLLM here)
        app_state.llm = "llama-2-7b"  # Placeholder
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down LLM API...")
    app_state.llm = None


# Create FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="Production-ready LLM API with FastAPI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to state
    request.state.request_id = request_id

    # Process request
    try:
        response = await call_next(request)

        # Calculate latency
        latency = (time.time() - start_time) * 1000

        # Log
        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"status={response.status_code} "
            f"latency_ms={latency:.2f}"
        )

        # Update stats
        app_state.total_requests += 1
        app_state.total_latency += latency

        return response

    except Exception as e:
        logger.error(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"error={str(e)}"
        )
        raise


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns service health status
    """
    uptime = time.time() - app_state.start_time

    avg_latency = (
        app_state.total_latency / app_state.total_requests
        if app_state.total_requests > 0
        else 0.0
    )

    return HealthResponse(
        status="healthy" if app_state.llm else "unhealthy",
        model_loaded=app_state.llm is not None,
        uptime_seconds=uptime,
        total_requests=app_state.total_requests,
        average_latency_ms=avg_latency
    )


# Completion endpoint
@app.post("/v1/completions", response_model=CompletionResponse, tags=["Completions"])
async def create_completion(
    request: CompletionRequest,
    req: Request
) -> CompletionResponse:
    """
    Create text completion

    Generate text completion from prompt
    """
    start_time = time.time()

    # Check if model loaded
    if not app_state.llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Simulate generation (in production, use vLLM)
        await asyncio.sleep(0.1)  # Simulate latency

        generated_text = "In circuits deep and logic bright,\nWhere silicon dreams take flight..."

        # Calculate usage
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return CompletionResponse(
            id=req.state.request_id,
            text=generated_text,
            model=app_state.llm,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            created_at=time.time(),
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


# Streaming completion endpoint
@app.post("/v1/completions/stream", tags=["Completions"])
async def create_completion_stream(
    request: CompletionRequest,
    req: Request
) -> StreamingResponse:
    """
    Create streaming completion

    Stream text completion token by token
    """
    if not request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stream parameter must be true"
        )

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        # Simulate token-by-token generation
        tokens = "In circuits deep and logic bright,\nWhere silicon dreams take flight...".split()

        for i, token in enumerate(tokens):
            # Simulate generation time
            await asyncio.sleep(0.05)

            chunk = {
                "id": req.state.request_id,
                "text": token + " ",
                "index": i,
                "finished": i == len(tokens) - 1
            }

            yield f"data: {json.dumps(chunk)}\n\n"

        # Send done signal
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint

    Returns API information
    """
    return {
        "name": "LLM Inference API",
        "version": "1.0.0",
        "model": app_state.llm,
        "status": "running",
        "docs": "/docs"
    }


# Run server
if __name__ == "__main__":
    print("="*80)
    print("STARTING FASTAPI SERVER")
    print("="*80)
    print("""
# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run production server (with workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (production)
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Swagger UI

curl -X POST http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Write a poem about AI",
    "max_tokens": 256,
    "temperature": 0.8
  }'

# Test streaming
curl -X POST http://localhost:8000/v1/completions/stream \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Write a poem",
    "max_tokens": 256,
    "stream": true
  }'
    """)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

## 2. Authentication

```python
"""
Authentication = Secure API access

Methods:
  1. API Keys:
     • Simple, stateless
     • Good for: Server-to-server
     • Format: Bearer <key>

  2. JWT Tokens:
     • Stateless, signed
     • Good for: User authentication
     • Contains: User ID, permissions, expiry

  3. OAuth2:
     • Industry standard
     • Good for: Third-party access
     • Complex but secure
"""

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import hashlib
import secrets
import jwt
from typing import Optional, Dict


# API Key authentication
class APIKeyAuth:
    """
    API Key authentication

    Example:
        >>> auth = APIKeyAuth()
        >>> key = auth.create_api_key(user_id="user123")
        >>> user = auth.verify_api_key(key)
    """

    def __init__(self):
        # In production, store in database
        self.api_keys: Dict[str, Dict] = {}

    def create_api_key(
        self,
        user_id: str,
        name: str = "default",
        rate_limit: int = 1000
    ) -> str:
        """
        Create new API key

        Args:
            user_id: User ID
            name: Key name
            rate_limit: Requests per hour

        Returns:
            API key
        """
        # Generate secure random key
        key = f"sk-{secrets.token_urlsafe(32)}"

        # Hash key for storage (never store plaintext!)
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Store metadata
        self.api_keys[key_hash] = {
            "user_id": user_id,
            "name": name,
            "rate_limit": rate_limit,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "total_requests": 0
        }

        return key

    def verify_api_key(self, key: str) -> Optional[Dict]:
        """
        Verify API key

        Args:
            key: API key to verify

        Returns:
            User info if valid, None otherwise
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        if key_hash in self.api_keys:
            # Update last used
            self.api_keys[key_hash]["last_used"] = datetime.utcnow()
            self.api_keys[key_hash]["total_requests"] += 1

            return self.api_keys[key_hash]

        return None

    def revoke_api_key(self, key: str) -> bool:
        """Revoke API key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        if key_hash in self.api_keys:
            del self.api_keys[key_hash]
            return True

        return False


# JWT authentication
class JWTAuth:
    """
    JWT token authentication

    Example:
        >>> auth = JWTAuth(secret_key="your-secret")
        >>> token = auth.create_token(user_id="user123")
        >>> user = auth.verify_token(token)
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(
        self,
        user_id: str,
        expires_delta: timedelta = timedelta(hours=24)
    ) -> str:
        """
        Create JWT token

        Args:
            user_id: User ID
            expires_delta: Token expiration time

        Returns:
            JWT token
        """
        expire = datetime.utcnow() + expires_delta

        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token

        Args:
            token: JWT token

        Returns:
            Payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None


# FastAPI dependency for authentication
security = HTTPBearer()
api_key_auth = APIKeyAuth()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict:
    """
    Verify API key dependency

    Usage:
        @app.get("/protected")
        async def protected_route(user: Dict = Depends(verify_api_key)):
            return {"user": user["user_id"]}
    """
    key = credentials.credentials

    user = api_key_auth.verify_api_key(key)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


# Example protected endpoint
@app.post("/v1/completions/protected", response_model=CompletionResponse, tags=["Completions"])
async def create_completion_protected(
    request: CompletionRequest,
    req: Request,
    user: Dict = Depends(verify_api_key)
) -> CompletionResponse:
    """
    Protected completion endpoint

    Requires valid API key
    """
    logger.info(f"Request from user: {user['user_id']}")

    # Same as create_completion but with auth
    # ... (implementation)

    return CompletionResponse(
        id=req.state.request_id,
        text="Protected completion...",
        model=app_state.llm,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        created_at=time.time(),
        latency_ms=0.0
    )


# Demo authentication
def demo_authentication():
    """Demo authentication setup"""
    print("\n" + "="*80)
    print("AUTHENTICATION SETUP")
    print("="*80)

    print("""
API KEY AUTHENTICATION:

# 1. Create API key
auth = APIKeyAuth()
api_key = auth.create_api_key(
    user_id="user123",
    name="production-key",
    rate_limit=1000
)

# 2. Use API key
curl -X POST http://localhost:8000/v1/completions/protected \\
  -H "Authorization: Bearer <api-key>" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello"}'


JWT AUTHENTICATION:

# 1. Create token
jwt_auth = JWTAuth(secret_key="your-secret-key")
token = jwt_auth.create_token(user_id="user123")

# 2. Use token
curl -X POST http://localhost:8000/v1/completions/protected \\
  -H "Authorization: Bearer <jwt-token>" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello"}'


BEST PRACTICES:

1. API Keys:
   ✅ Never store plaintext (hash with SHA256)
   ✅ Use secure random generation (secrets.token_urlsafe)
   ✅ Prefix keys (sk- for secret, pk- for public)
   ✅ Store in database (not in-memory)
   ✅ Allow key rotation
   ✅ Track usage per key

2. JWT Tokens:
   ✅ Use strong secret key
   ✅ Set reasonable expiration (1-24 hours)
   ✅ Include minimal data in payload
   ✅ Validate signature and expiry
   ✅ Use HTTPS only

3. Environment Variables:
   # .env file
   SECRET_KEY=your-super-secret-key-change-this
   JWT_SECRET=your-jwt-secret
   API_KEY_HASH_SALT=random-salt

   # Load with python-dotenv
   from dotenv import load_dotenv
   load_dotenv()

4. Rate Limiting:
   ✅ Per API key limits
   ✅ Global limits
   ✅ Exponential backoff on abuse
   ✅ See next section for implementation
    """)


if __name__ == "__main__":
    demo_authentication()
```

*[Suite avec Rate Limiting et Monitoring dans la partie 2...]*

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables early (before importing local services)
load_dotenv()

# Local imports
from .core.middleware import RateLimitMiddleware
from .routers.chat import router as chat_router
from .routers.cases import router as cases_router
from .routers.health import router as health_router
from .routers.upload import router as upload_router
from .services.rag import init_rag_service, shutdown_rag_service
from .services.cases import init_cases_service, shutdown_cases_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ],
)
logger = logging.getLogger(__name__)

 

app = FastAPI(title="NyaySarthi Backend", version="1.0.0")

# Rate limiting
app.add_middleware(RateLimitMiddleware, max_requests=100, window=60)

# CORS configuration
origins_env = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").strip()
if origins_env == "*":
    cors_allow_origins = ["*"]
    cors_allow_credentials = False
else:
    cors_allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Authorization", "Cache-Control", "Pragma", "X-Requested-With", "x-request-id"],
    expose_headers=["*"],
    max_age=600,
)

# Routers
app.include_router(health_router, tags=["health"])  # basic health first
app.include_router(chat_router, tags=["chat"])      # /chat
app.include_router(cases_router, tags=["cases"])    # /cases/search
app.include_router(upload_router, tags=["upload"])  # /api/analyze-document

# Startup / Shutdown
@app.on_event("startup")
async def on_startup():
    # Initialize services
    # RAG can fail if models/embeddings cannot be downloaded. Do not block the app.
    try:
        await init_rag_service()
    except Exception as e:
        logging.warning(f"[STARTUP] RAG init failed, continuing without RAG: {e}")
    # Cases service should be light; still guard to avoid full crash
    try:
        await init_cases_service()
    except Exception as e:
        logging.error(f"[STARTUP] Cases service init failed: {e}")
        # Do not raise here to keep the server alive for other routes

@app.on_event("shutdown")
async def on_shutdown():
    # Shutdown services
    await shutdown_rag_service()
    await shutdown_cases_service()

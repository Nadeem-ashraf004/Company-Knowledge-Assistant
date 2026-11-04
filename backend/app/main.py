from fastapi import FastAPI, Request
import time
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger , setup_logging

from app.api.routes import auth, documents, chat, health

setup_logging()

app = FastAPI(
    title="Company Knowledge Assistant",
    description="Production-grade RAG-based company knowledge assistant",
    version="1.0.0",
)

# request logging middleware 
@app.middleware("http")
async def log_requests(request:Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception :
        logger.exception(
            "Unhandled exception occurred while processing request | method=%s path=%s ",
            request.method,
            request.url.path,
        ) 
        raise
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "HTTP request | method=%s path=%s duration=%.2f ms",
        request.method,
        request.url.path,
        duration_ms
    )
    return response
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# Register API routes
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])


@app.get("/")
async def root():
    return {
        "message": "Company Knowledge Assistant API",
        "status": "running",
        "version": "1.0.0",
    }
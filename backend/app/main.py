import time


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger , setup_logging

from app.api.routes import auth, documents, chat, health
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

setup_logging()

app = FastAPI(
    title="Company Knowledge Assistant",
    description="Production-grade RAG-based company knowledge assistant",
    version="1.0.0",
)
#centrilize exception handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "UnExpected application error | method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail":" An unexppected error occurred . please contact support if the issue persists."
        },
    )

# request logging middleware 
@app.middleware("http")
async def log_requests(request:Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception :
        logger.exception(
            "Unhandled request exception | method=%s path=%s ",
            request.method,
            request.url.path,
            
        ) 
        raise
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "HTTP request | method=%s path=%s duration=%.2f ms",
        request.method,
        request.url.path,
        response.status_code,
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
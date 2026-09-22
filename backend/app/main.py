from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, documents, chat, health

from app.db.database import init_db
app = FastAPI(
    title="Company Knowledge Assistant",
    description="Production-grade RAG-based company knowledge assistant",
    version="1.0.0",
)

init_db()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We will restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
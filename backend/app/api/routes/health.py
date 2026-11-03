from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.db.database import engine
from app.rag.vector_store import get_qdrant_client



router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    # check whether the application running and the database is reachable
    return {
        "status": "healthy",
        "service": "Company Knowledge Assistant",
    }
@router.get("/health/ready")
def readiness_check():
    # check whether the external services are reachable (e.g., database, vector store)
    checks={
        "database": "unavalaible",
        "qdrant": "unavalaible",
    }
    # chack the postgres database connection
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        checks["database"] = "available"
    except Exception :
        pass

    # check the Qdrant vector store connection
    client = None
    try:
        qdrant_client = get_qdrant_client()
        qdrant_client.health_check()
        checks["qdrant"] = "available"
    except Exception :
        pass 

    finally:
        if client is not None:
            client.close()

    all_available = all(
        status == "available"
        for status in checks.values()
    )    

    response = {
        "status" : "ready" if all_available else "not ready",
        "services" : "company knowlege assistant",
        "checks" : checks
    }

    return JSONResponse(
        status_code=200 if all_available else 503,
        content=checks
        )
from fastapi import APIRouter


router = APIRouter()


@router.post("/register")
async def register():
    return {
        "message": "Registration endpoint - implementation coming soon"
    }


@router.post("/login")
async def login():
    return {
        "message": "Login endpoint - implementation coming soon"
    }
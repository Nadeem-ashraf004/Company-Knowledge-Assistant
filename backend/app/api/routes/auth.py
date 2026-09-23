from fastapi import APIRouter ,Depends , HTTPException , status
from sqlalchemy.orm import Session
from app.schemas.user import UserResponse


from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)

from app.db.crud import create_record, get_user_by_email

from app.db.database import get_db

from app.models.user import User

from app.schemas.user import(
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)


router = APIRouter()


@router.post("/register" , response_model=UserResponse , status_code=status.HTTP_201_CREATED)
async def register(
    user_data = UserCreate,
    db:Session = Depends(get_db)
)->UserResponse:
    # register a new user
    # check wether the user already reistered
    existing_user = get_user_by_email(db, email = user_data.email)
    if existing_user:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail = "user with this email already exists",
        )
    # hash the password before storing it in database
    hashed_password = hash_password(user_data.password)
    #create a databsase 
    user=User(
        email = user_data.email,
        password_hash = hashed_password,
        is_active = True
    )

    user = create_record(
        db=db,
        record = user,
        )
    

    return user

@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK     
        )
async def login(
    user_data : UserLogin,
    db: Session = Depends(get_db),
) -> TokenResponse:
    # authentication  a user  and return a JWT token.
    # find the user by email
    user = get_user_by_email(
        db= db,
        email= user_data.email,
    )


    #use the verify password function to check if the provided password matches the stored hash
    # to avoid revealing wether an accout exist
    if not user or not verify_password(
        user_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid email or password",
            headers={"WWW-Authentucation":"Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user account is inactive",
        )
    #store the user id in the jwt token payload
    access_token = create_access_token(
        data={
            "sub": str(user.id),
        }
    )

    return TokenResponse(
        access_token=access_token,
        token_type = "bearer",  
            )
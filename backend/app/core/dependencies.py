from uuid import UUID
from fastapi import Depends , HTTPException , status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from app.core.security import decode_access_token, verify_password
from app.db.crud import get_user_by_id
from app.db.database import get_db
from app.models.user import User


#extract the token from thr authorization header
#authorization header is in the formate : baerer <token>
security = HTTPBearer(auto_error=True)

def get_current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
        db: Session = Depends(get_db),
) -> User:
    #  Authenticate the request and return the current active user .
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authentication":"Bearer"},
        )
    # decode the token and get the user id
    payload = decode_access_token(credentials.credentials)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate":"Bearer"},
        )
    # extract the user id form the payload
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incalid takon payload",
            headers={"WWW-AUthentication ": "Bearer"},
        )
    #conver the jwt subject into a uuid object
    try:
        user_id = UUID(subject)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token subject",
            headers={"WWW-Authentication":"Bearer"},
        )   
    #load the user from the postgres database using the user id
    user = get_user_by_id(
        db=db,
        user_id=user_id,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='user not found',
            headers={"WWW-Authentication":"Bearer"},
        )
    #prevent inactive users the api
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user account is inactive",
            headers={"WWW-Authentication":"Bearer"},
        )
    return user


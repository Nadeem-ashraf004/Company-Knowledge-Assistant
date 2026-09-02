from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.user import User
from app.schemas.user import UserCreate


def get_user_by_email(
    db: Session,
    email: str,
) -> User | None:
    """Find a user by email address."""

    statement = select(User).where(User.email == email)

    return db.scalar(statement)


def get_user_by_id(
    db: Session,
    user_id,
) -> User | None:
    """Find a user by ID."""

    return db.get(User, user_id)


def create_user(
    db: Session,
    user_data: UserCreate,
) -> User:
    """Create a new user with a hashed password."""

    existing_user = get_user_by_email(
        db,
        user_data.email,
    )

    if existing_user:
        raise ValueError("User with this email already exists")

    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_user(
    db: Session,
    email: str,
    password: str,
) -> User | None:
    """Authenticate a user using email and password."""

    user = get_user_by_email(
        db,
        email,
    )

    if not user:
        return None

    if not user.is_active:
        return None

    if not verify_password(
        password,
        user.password_hash,
    ):
        return None

    return user
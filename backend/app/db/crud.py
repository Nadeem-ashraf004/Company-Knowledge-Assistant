from typing import Any
from uuid import UUID
from huggingface_hub import User
from sqlalchemy.orm import Session


def create_record(
    db: Session,
    record: Any,
) -> Any:
    """Create and save a database record."""

    db.add(record)
    db.commit()
    db.refresh(record)

    return record


def delete_record(
    db: Session,
    record: Any,
) -> None:
    """Delete a database record."""

    db.delete(record)
    db.commit()



def get_user_by_email(
        db: Session,
        email: str,
) -> User | None:
    # fined use by email
    statement =  select(User).where(User.email == email)

    return db.scalar(statement)

def get_user_by_id(
        db : Session,
        user_id : UUID,
) -> User | None :
    # find user by id

    statement = select(User).where(user_id == User.id)



    return db.scalar(statement)   
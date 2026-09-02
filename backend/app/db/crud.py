from typing import Any

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
"""
@filename: eval_db_manager.py
@description: Evaluation DB ORM models and CRUD operations (fixed slots)
"""
from typing import List, Optional
from sqlalchemy import Column, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

try:
    from .eval_db_config import engine as default_engine
    from .eval_db_config import SessionLocal as DefaultSessionLocal
except ImportError:
    from eval_db_config import engine as default_engine
    from eval_db_config import SessionLocal as DefaultSessionLocal

Base = declarative_base()

MAX_EVALS = 15


class Evaluation(Base):
    """One row per sample_name, with fixed evaluation slots."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_name = Column(String(255), nullable=False)

    model_name1 = Column(String(255))
    model_name2 = Column(String(255))
    model_name3 = Column(String(255))
    model_name4 = Column(String(255))
    model_name5 = Column(String(255))
    model_name6 = Column(String(255))
    model_name7 = Column(String(255))
    model_name8 = Column(String(255))
    model_name9 = Column(String(255))
    model_name10 = Column(String(255))
    model_name11 = Column(String(255))
    model_name12 = Column(String(255))
    model_name13 = Column(String(255))
    model_name14 = Column(String(255))
    model_name15 = Column(String(255))

    eval_text1 = Column(Text)
    eval_text2 = Column(Text)
    eval_text3 = Column(Text)
    eval_text4 = Column(Text)
    eval_text5 = Column(Text)
    eval_text6 = Column(Text)
    eval_text7 = Column(Text)
    eval_text8 = Column(Text)
    eval_text9 = Column(Text)
    eval_text10 = Column(Text)
    eval_text11 = Column(Text)
    eval_text12 = Column(Text)
    eval_text13 = Column(Text)
    eval_text14 = Column(Text)
    eval_text15 = Column(Text)

    __table_args__ = (
        UniqueConstraint("sample_name", name="uq_sample_name"),
        Index("ix_evaluations_sample_name", "sample_name"),
    )


def _model_col(i: int) -> str:
    return f"model_name{i}"


def _text_col(i: int) -> str:
    return f"eval_text{i}"


def _iter_slots(row: Evaluation):
    for i in range(1, MAX_EVALS + 1):
        model_name = getattr(row, _model_col(i))
        eval_text = getattr(row, _text_col(i))
        yield i, model_name, eval_text


def _is_empty(val) -> bool:
    return val is None or (isinstance(val, str) and val.strip() == "")


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(default_engine)


def add_or_update_evaluation(
    sample_name: str,
    model_name: str,
    eval_text: str,
) -> None:
    """
    Insert or update an evaluation.
    - If model_name exists in a slot, update its eval_text.
    - Else, put it into the first empty slot.
    """
    session = DefaultSessionLocal()
    try:
        row = (
            session.query(Evaluation)
            .filter(Evaluation.sample_name == sample_name)
            .first()
        )
        if row is None:
            row = Evaluation(sample_name=sample_name)
            session.add(row)
            session.flush()

        # Update existing slot if model_name matches
        for i, existing_model, _ in _iter_slots(row):
            if existing_model == model_name:
                setattr(row, _text_col(i), eval_text)
                session.commit()
                return

        # Otherwise, insert into first empty slot
        for i, existing_model, _ in _iter_slots(row):
            if _is_empty(existing_model):
                setattr(row, _model_col(i), model_name)
                setattr(row, _text_col(i), eval_text)
                session.commit()
                return

        raise ValueError(f"No empty evaluation slots for sample '{sample_name}'.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _evaluation_to_dict(row: Evaluation) -> dict:
    data = {"id": row.id, "sample_name": row.sample_name}
    for i, model_name, eval_text in _iter_slots(row):
        data[_model_col(i)] = model_name
        data[_text_col(i)] = eval_text
    return data


def get_evaluation(
    sample_name: str,
    model_name: str,
) -> Optional[dict]:
    """Get a single evaluation by sample_name and model_name."""
    session = DefaultSessionLocal()
    try:
        row = (
            session.query(Evaluation)
            .filter(Evaluation.sample_name == sample_name)
            .first()
        )
        if row is None:
            return None

        for i, existing_model, eval_text in _iter_slots(row):
            if existing_model == model_name:
                return {
                    "id": row.id,
                    "sample_name": row.sample_name,
                    "model_name": existing_model,
                    "eval_text": eval_text,
                    "slot_index": i,
                }

        return None
    finally:
        session.close()


def get_evaluations(sample_name: str) -> List[dict]:
    """Get all evaluations for a sample_name."""
    session = DefaultSessionLocal()
    try:
        row = (
            session.query(Evaluation)
            .filter(Evaluation.sample_name == sample_name)
            .first()
        )
        if row is None:
            return []

        results = []
        for i, model_name, eval_text in _iter_slots(row):
            if _is_empty(model_name):
                continue
            results.append(
                {
                    "id": row.id,
                    "sample_name": row.sample_name,
                    "model_name": model_name,
                    "eval_text": eval_text,
                    "slot_index": i,
                }
            )
        return results
    finally:
        session.close()


def list_samples(limit: Optional[int] = None, offset: int = 0) -> List[dict]:
    """List sample_name values."""
    session = DefaultSessionLocal()
    try:
        query = session.query(Evaluation).order_by(Evaluation.sample_name.asc())
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return [{"id": row.id, "sample_name": row.sample_name} for row in query.all()]
    finally:
        session.close()


def rename_model(
    sample_name: str,
    old_model_name: str,
    new_model_name: str,
) -> int:
    """Rename a model entry for a sample. Returns number of rows updated."""
    if old_model_name == new_model_name:
        return 0

    session = DefaultSessionLocal()
    try:
        row = (
            session.query(Evaluation)
            .filter(Evaluation.sample_name == sample_name)
            .first()
        )
        if row is None:
            return 0

        # Reject if new name already exists
        for _, existing_model, _ in _iter_slots(row):
            if existing_model == new_model_name:
                raise ValueError(
                    f"Model '{new_model_name}' already exists for sample '{sample_name}'."
                )

        for i, existing_model, _ in _iter_slots(row):
            if existing_model == old_model_name:
                setattr(row, _model_col(i), new_model_name)
                session.commit()
                return 1

        return 0
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def delete_evaluation(sample_name: str, model_name: str) -> int:
    """Delete a single evaluation slot by model_name."""
    session = DefaultSessionLocal()
    try:
        row = (
            session.query(Evaluation)
            .filter(Evaluation.sample_name == sample_name)
            .first()
        )
        if row is None:
            return 0

        for i, existing_model, _ in _iter_slots(row):
            if existing_model == model_name:
                setattr(row, _model_col(i), None)
                setattr(row, _text_col(i), None)

                # If all empty, remove the row entirely
                if all(_is_empty(m) for _, m, _ in _iter_slots(row)):
                    session.delete(row)
                session.commit()
                return 1

        return 0
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def delete_sample(sample_name: str) -> int:
    """Delete a sample row. Returns 1 if deleted, else 0."""
    session = DefaultSessionLocal()
    try:
        row = (
            session.query(Evaluation)
            .filter(Evaluation.sample_name == sample_name)
            .first()
        )
        if row is None:
            return 0

        session.delete(row)
        session.commit()
        return 1
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

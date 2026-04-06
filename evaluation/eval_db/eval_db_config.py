"""
@filename: eval_db_config.py
@description: Evaluation DB config (SQLAlchemy engine/session)
"""
import os
import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def _enable_sqlite_fk(db_engine: Engine) -> None:
    @event.listens_for(db_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

DEFAULT_DB_FILENAME = "evaluation3.db"
env_db_path = os.getenv("SEMA_EVAL_DB_PATH", "").strip()
if env_db_path:
    db_path = env_db_path if os.path.isabs(env_db_path) else os.path.join(script_dir, env_db_path)
else:
    db_path = os.path.join(script_dir, DEFAULT_DB_FILENAME)

DATABASE_URL = "sqlite:///" + db_path
engine = create_engine(DATABASE_URL, echo=False)
_enable_sqlite_fk(engine)

SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

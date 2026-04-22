"""Database connection and session management for PostgreSQL (Neon serverless)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

load_dotenv()

_engine_singleton: Any = None


class Base(DeclarativeBase):
    pass


def create_db_engine(*, connect_args: dict[str, Any] | None = None) -> Any:
    url = os.environ.get("DATABASE_URL")
    if not url:
        msg = "DATABASE_URL environment variable is not set"
        raise RuntimeError(msg)

    default_connect_args: dict[str, Any] = {"sslmode": "require"}
    if connect_args:
        default_connect_args.update(connect_args)

    return create_engine(
        url,
        connect_args=default_connect_args,
        pool_size=5,
        max_overflow=10,
        # Neon suspends idle computes; recycle before the connection goes stale
        pool_recycle=300,
        pool_pre_ping=True,
    )


def get_engine() -> Any:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = create_db_engine()
    return _engine_singleton


def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

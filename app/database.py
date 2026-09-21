"""
Database setup. SQLite by default (zero-config). Swap SQLALCHEMY_DATABASE_URL
for Postgres/MySQL in production without touching the rest of the app.
"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = "sqlite:///./goldsignal.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_columns() -> None:
    """Tiny auto-migration: SQLAlchemy's create_all() never adds columns to a table
    that already exists, so add any missing ones (keeps your existing goldsignal.db)."""
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have:
                ddl = col.type.compile(dialect=engine.dialect)
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

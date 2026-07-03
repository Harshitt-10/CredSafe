from pathlib import Path
import sqlite3

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "data" / "home_credit.db"
SQL_PATH = PROJECT_ROOT / "sql" / "create_views.sql"


def create_connection():
    """Create and return a SQLite database connection."""
    return sqlite3.connect(DB_PATH)


def execute_sql_file(conn, sql_file):
    """Execute all SQL statements from a .sql file."""
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    sql_script = sql_file.read_text(encoding="utf-8")

    conn.executescript(sql_script)
    conn.commit()


def main():
    print("Creating SQL views...")

    conn = create_connection()

    try:
        execute_sql_file(conn, SQL_PATH)
        print("SQL views created successfully!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

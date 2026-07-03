from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "home_credit.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        ORDER BY type, name;
    """)

    objects = cursor.fetchall()

    print("\nDatabase Objects:\n")

    for name, obj_type in objects:
        print(f"{obj_type:<5} : {name}")

    conn.close()


if __name__ == "__main__":
    main()

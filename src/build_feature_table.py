from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "data" / "home_credit.db"
QUERY_PATH = PROJECT_ROOT / "sql" / "final_query.sql"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_table.csv"


def create_connection():
    return sqlite3.connect(DB_PATH)


def main():
    conn = create_connection()

    query = QUERY_PATH.read_text(encoding="utf-8")

    print("Building feature table...")

    feature_df = pd.read_sql_query(query, conn)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature table created successfully.")
    print(f"Rows    : {len(feature_df):,}")
    print(f"Columns : {feature_df.shape[1]}")
    print(f"Saved to: {OUTPUT_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
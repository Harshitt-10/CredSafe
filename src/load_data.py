from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "home_credit.db"

TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
}


def create_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = OFF;")
    return conn


def load_table(conn, table_name, filename, chunksize=50000):
    csv_path = RAW_DATA_DIR / filename

    print(f"Loading {table_name}...")

    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk.to_sql(
            table_name,
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )

        total_rows += len(chunk)
        print(f"  Processed {total_rows:,} rows", end="\r")

    print(f"Loaded {table_name}: {total_rows:,} rows")


def main():
    conn = create_connection()

    for table_name, filename in TABLES.items():
        load_table(conn, table_name, filename)

    conn.close()
    print("\nDatabase created successfully.")


if __name__ == "__main__":
    main()

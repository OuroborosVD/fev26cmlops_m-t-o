from pathlib import Path
import pandas as pd
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "weatherAUS_encoded.csv"
DB_PATH = DATA_DIR / "weather.db"
TABLE_NAME = "weather_data"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print(f"Données chargées dans la base : {DB_PATH}")
    print(f"Table créée : {TABLE_NAME}")
    print(f"Nombre de lignes : {len(df)}")


if __name__ == "__main__":
    main()
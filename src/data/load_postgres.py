from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# --- PATH DATA ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "weatherAUS_encoded.csv"

# --- POSTGRES CONFIG ---
DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "weather_db"

TABLE_NAME = "weather_data"


def get_engine():
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Shape: {df.shape}")

    engine = get_engine()

    print("Writing to PostgreSQL...")
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
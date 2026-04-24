import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


# --- CHEMINS --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "weatherAUS_encoded.csv"


# --- CONFIGURATION POSTGRESQL ----------------------------------------------------------

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "weather_db")

TABLE_NAME = "weather_data"


def get_engine():
    """Crée la connexion SQLAlchemy vers PostgreSQL."""
    database_url = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(database_url)


def load_dataset():
    """Charge le dataset retraité utilisé comme source de référence du projet."""
    print("Chargement du dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dimensions du dataset : {df.shape}")
    return df


def write_to_postgres(df):
    """Remplace la table PostgreSQL par la version courante du dataset."""
    print("Connexion à PostgreSQL...")
    engine = get_engine()

    print(f"Écriture dans la table PostgreSQL : {TABLE_NAME}")
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print("Chargement terminé avec succès.")


def main():
    df = load_dataset()
    write_to_postgres(df)


if __name__ == "__main__":
    main()
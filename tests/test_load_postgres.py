# Importation de modules
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data.load_postgres import load_dataset, write_to_postgres


@patch("src.data.load_postgres.pd.read_csv")
def test_load_dataset(mock_read_csv):
    """ Test du chargement du dataset sans lire de fichier réel """

    # --- Simulation du DataFrame retourné ---
    fake_df = pd.DataFrame({
        "temp": [20, 25],
        "rain": [0, 1]
    })

    mock_read_csv.return_value = fake_df  
    result = load_dataset()

    assert result.equals(fake_df)
    mock_read_csv.assert_called_once()  # vérifie que read_csv a été appelé


@patch("src.data.load_postgres.get_engine")
def test_write_to_postgres(mock_get_engine):
    """ Test de l'écriture vers PostgreSQL sans vraie base de données """

    # --- Faux DataFrame ---
    df = pd.DataFrame({
        "temp": [20, 25],
        "rain": [0, 1]
    })

    # --- Mock engine ---
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    # --- On mock aussi la méthode to_sql du DataFrame ---
    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:

        write_to_postgres(df)

        # Vérifie que l'engine a été créé
        mock_get_engine.assert_called_once()

        # Vérifie que to_sql a été appelé correctement
        mock_to_sql.assert_called_once()

        # Vérifie les arguments importants de to_sql
        args, kwargs = mock_to_sql.call_args

        assert args[0] == "weather_data"  # nom de table
        assert kwargs["if_exists"] == "replace"
        assert kwargs["index"] is False

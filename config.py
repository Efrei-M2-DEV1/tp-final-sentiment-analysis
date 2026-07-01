"""Configuration centralisée du projet.

Charge les variables d'environnement (fichier .env) et expose les
constantes utilisées par l'API, la base de données et les modèles.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Charge le fichier .env s'il existe (sinon on garde les valeurs par défaut).
load_dotenv()

# --- Arborescence du projet ---------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Chemins des deux modèles (positif / négatif) et du vectoriseur partagé.
POSITIVE_MODEL_PATH = MODELS_DIR / "model_positive.pkl"
NEGATIVE_MODEL_PATH = MODELS_DIR / "model_negative.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

# --- Base de données MySQL ----------------------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}
DB_NAME = os.getenv("DB_NAME", "socialmetrics")

# --- API Flask -----------------------------------------------------------
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

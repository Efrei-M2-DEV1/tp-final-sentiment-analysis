"""Réentraînement hebdomadaire des modèles.

Ce script est destiné à être exécuté périodiquement (voir scripts/crontab.txt
sous Linux/macOS, ou le Planificateur de tâches Windows). Il relance
l'entraînement sur les données les plus récentes de la table `tweets` et
journalise le résultat dans logs/retrain.log.

Usage manuel :
    python scripts/retrain.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Rend les modules du projet importables.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import train  # noqa: E402

# Journalisation dans un fichier (créé à la volée).
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "retrain.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    start = datetime.now()
    logging.info("Début du réentraînement.")
    try:
        train()
    except Exception:  # on trace l'erreur complète dans le log
        logging.exception("Échec du réentraînement.")
        raise
    duration = (datetime.now() - start).total_seconds()
    logging.info("Réentraînement terminé en %.1f s.", duration)
    print(f"Réentraînement terminé (voir {LOG_DIR / 'retrain.log'}).")


if __name__ == "__main__":
    main()

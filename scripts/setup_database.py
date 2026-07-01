"""Script de configuration de la base de données.

Usage :
    python scripts/setup_database.py

Crée la base MySQL, la table `tweets` puis insère le jeu de données
d'amorçage si la table est vide.
"""
import sys
from pathlib import Path

# Permet d'importer les modules du projet depuis n'importe quel dossier.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tweets_seed import SEED_TWEETS  # noqa: E402
from src.database import count_tweets, init_database, insert_tweets  # noqa: E402


def main() -> None:
    print("Création de la base de données et de la table `tweets`...")
    init_database()
    print("Base et table prêtes.")

    existing = count_tweets()
    if existing > 0:
        print(f"La table contient déjà {existing} tweets. Aucun ajout.")
        return

    inserted = insert_tweets(SEED_TWEETS)
    print(f"{inserted} tweets d'amorçage insérés.")


if __name__ == "__main__":
    main()

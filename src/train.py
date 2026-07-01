"""Entraînement des deux modèles de régression logistique.

Ce script :
  1. charge les tweets annotés depuis la base MySQL ;
  2. nettoie puis vectorise les textes (Bag-of-Words) ;
  3. entraîne DEUX régressions logistiques (positif / négatif) ;
  4. évalue chaque modèle (rapport + matrice de confusion) ;
  5. sauvegarde le vectoriseur et les deux modèles sur disque.

Usage :
    python src/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permet de lancer le script directement (python src/train.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import NEGATIVE_MODEL_PATH, POSITIVE_MODEL_PATH, VECTORIZER_PATH
from src.database import fetch_tweets
from src.preprocessing import build_vectorizer, clean_corpus


def _train_one(name: str, X_train, X_test, y_train, y_test) -> LogisticRegression:
    """Entraîne et évalue un modèle de régression logistique.

    Args:
        name: libellé du modèle (« positif » / « négatif ») pour l'affichage.
        X_train, X_test: matrices Bag-of-Words.
        y_train, y_test: labels binaires (0/1).

    Returns:
        Le modèle entraîné.
    """
    # class_weight="balanced" compense le déséquilibre des classes
    # (il y a plus de tweets « non positifs » que « positifs », etc.).
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n=== Modèle « {name} » ===")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    return model


def train() -> None:
    """Pipeline complet d'entraînement des deux modèles."""
    df = fetch_tweets()
    if df.empty:
        raise SystemExit(
            "La table `tweets` est vide. Lancez d'abord : "
            "python scripts/setup_database.py"
        )

    print(f"{len(df)} tweets chargés depuis la base.")

    # 1. Nettoyage + vectorisation (le vectoriseur est PARTAGÉ par les 2 modèles).
    texts = clean_corpus(df["text"].tolist())
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    y_pos = df["positive"].to_numpy()
    y_neg = df["negative"].to_numpy()

    # 2. Découpe train/test unique (mêmes indices pour comparer les 2 modèles).
    indices = range(X.shape[0])
    idx_train, idx_test = train_test_split(
        list(indices), test_size=0.25, random_state=42, stratify=y_pos
    )
    X_train, X_test = X[idx_train], X[idx_test]

    # 3. Entraînement des deux modèles.
    model_pos = _train_one(
        "positif", X_train, X_test, y_pos[idx_train], y_pos[idx_test]
    )
    model_neg = _train_one(
        "négatif", X_train, X_test, y_neg[idx_train], y_neg[idx_test]
    )

    # 4. Sauvegarde des artefacts.
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model_pos, POSITIVE_MODEL_PATH)
    joblib.dump(model_neg, NEGATIVE_MODEL_PATH)
    print(f"\nArtefacts sauvegardés dans : {VECTORIZER_PATH.parent}")


if __name__ == "__main__":
    train()

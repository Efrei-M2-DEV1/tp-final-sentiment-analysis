"""Chargement des modèles entraînés et calcul du score de sentiment.

Le prototype repose sur DEUX modèles de régression logistique distincts :
  - `model_positive`  : prédit la probabilité qu'un tweet soit positif ;
  - `model_negative`  : prédit la probabilité qu'un tweet soit négatif.

Pourquoi deux modèles et non un seul ? Un tweet peut être positif, négatif
OU neutre (ni l'un ni l'autre). Un unique classifieur binaire ne saurait pas
représenter le cas neutre. En combinant les deux probabilités, on obtient un
score continu entre -1 (très négatif) et +1 (très positif), 0 étant neutre.

    score = P(positif) - P(négatif)
"""
from __future__ import annotations

import joblib

from config import NEGATIVE_MODEL_PATH, POSITIVE_MODEL_PATH, VECTORIZER_PATH
from preprocessing import clean_text


class SentimentAnalyzer:
    """Charge le vectoriseur et les deux modèles, puis calcule les scores.

    Les artefacts (`.pkl`) sont produits par `src/train.py`. On les charge une
    seule fois à l'instanciation, ce qui évite de relire le disque à chaque
    requête de l'API (coûteux) — les prédictions se font ensuite en mémoire.
    """

    def __init__(self) -> None:
        self.vectorizer = None
        self.model_positive = None
        self.model_negative = None
        self.load()

    def load(self) -> None:
        """Charge les artefacts depuis le disque.

        Raises:
            FileNotFoundError: si un modèle n'a pas encore été entraîné.
        """
        for path in (VECTORIZER_PATH, POSITIVE_MODEL_PATH, NEGATIVE_MODEL_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Artefact manquant : {path}. "
                    "Lancez d'abord l'entraînement : python src/train.py"
                )
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.model_positive = joblib.load(POSITIVE_MODEL_PATH)
        self.model_negative = joblib.load(NEGATIVE_MODEL_PATH)

    def _proba(self, model, matrix):
        """Renvoie la probabilité de la classe positive (label 1) du modèle.

        `predict_proba` renvoie deux colonnes [P(0), P(1)] ; on récupère P(1).
        Si le modèle n'a été entraîné que sur une seule classe (cas dégénéré
        d'un très petit jeu de données), on retombe sur cette classe.
        """
        proba = model.predict_proba(matrix)
        classes = list(model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        # Le modèle ne connaît que la classe 0 -> probabilité de 1 nulle.
        return proba[:, 0] * 0.0

    def score_tweets(self, tweets: list[str]) -> list[float]:
        """Calcule un score de sentiment dans [-1, 1] pour chaque tweet.

        Args:
            tweets: liste de tweets bruts.

        Returns:
            Liste de scores (float) alignée sur `tweets`.
        """
        if not tweets:
            return []
        cleaned = [clean_text(t) for t in tweets]
        matrix = self.vectorizer.transform(cleaned)
        p_pos = self._proba(self.model_positive, matrix)
        p_neg = self._proba(self.model_negative, matrix)
        # score = P(positif) - P(négatif), borné dans [-1, 1] par construction.
        return [round(float(pp - pn), 4) for pp, pn in zip(p_pos, p_neg)]

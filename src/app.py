"""API REST Flask d'analyse de sentiments.

Expose un endpoint POST /analyze qui accepte une liste de tweets et renvoie,
pour chacun, un score de sentiment entre -1 (très négatif) et +1 (très
positif) au format JSON.

Exemple de requête :
    POST /analyze
    { "tweets": ["J'adore ce produit", "Service catastrophique"] }

Exemple de réponse :
    {
      "J'adore ce produit": 0.82,
      "Service catastrophique": -0.79
    }

Usage :
    python src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Rend les modules du projet importables quel que soit le dossier de lancement.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request

from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from model import SentimentAnalyzer

app = Flask(__name__)

# Le modèle est chargé paresseusement : on ne bloque pas le démarrage de l'API
# si les artefacts ne sont pas encore présents (message d'erreur explicite).
_analyzer: SentimentAnalyzer | None = None


def get_analyzer() -> SentimentAnalyzer:
    """Renvoie l'analyseur, en le chargeant à la première utilisation."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


@app.get("/health")
def health():
    """Vérifie que l'API répond (utile pour la supervision)."""
    return jsonify({"status": "ok"})


@app.post("/analyze")
def analyze():
    """Analyse une liste de tweets et renvoie un score par tweet.

    Gère explicitement les erreurs demandées par l'énoncé : corps JSON
    absent, champ manquant, type incorrect, liste vide.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Corps JSON attendu (Content-Type application/json)."}), 400

    tweets = data.get("tweets")
    if tweets is None:
        return jsonify({"error": "Champ 'tweets' manquant."}), 400
    if not isinstance(tweets, list):
        return jsonify({"error": "'tweets' doit être une liste de chaînes."}), 400
    if len(tweets) == 0:
        return jsonify({"error": "La liste 'tweets' est vide."}), 400
    if not all(isinstance(t, str) for t in tweets):
        return jsonify({"error": "Chaque tweet doit être une chaîne de caractères."}), 400

    try:
        analyzer = get_analyzer()
    except FileNotFoundError as exc:
        # Les modèles n'ont pas encore été entraînés.
        return jsonify({"error": str(exc)}), 503

    scores = analyzer.score_tweets(tweets)
    # Réponse au format demandé : { "tweet": score, ... }.
    return jsonify({tweet: score for tweet, score in zip(tweets, scores)})


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

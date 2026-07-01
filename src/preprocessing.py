"""Prétraitement du texte des tweets.

Reprend la démarche vue en cours / dans l'exercice « développer un modèle
IA » : mise en minuscules, suppression des caractères spéciaux, retrait des
mots vides (stop-words) et vectorisation Bag-of-Words.
"""
import re

# Liste simple de mots vides en français (peu informatifs pour la classification).
FRENCH_STOPWORDS = [
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "et", "en", "au",
    "aux", "avec", "ce", "ces", "pour", "par", "sur", "pas", "plus", "où", "mais",
    "ou", "donc", "ni", "car", "ne", "que", "qui", "quoi", "quand", "à", "son",
    "sa", "ses", "ils", "elles", "nous", "vous", "est", "sont", "cette", "cet",
    "aussi", "être", "avoir", "faire", "comme", "tout", "bien", "on", "lui",
    "je", "tu", "il", "elle", "se", "me", "te", "y", "ai", "as", "a", "ont",
]

# Pré-compilation de l'expression régulière (évite de la recompiler à chaque appel).
_SPECIAL_CHARS = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean_text(text: str) -> str:
    """Nettoie un texte : minuscules + suppression des caractères spéciaux.

    Args:
        text: le tweet brut.

    Returns:
        Le texte nettoyé, prêt à être vectorisé.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _SPECIAL_CHARS.sub(" ", text)
    # Normalise les espaces multiples.
    return re.sub(r"\s+", " ", text).strip()


def build_vectorizer(max_features: int = 500) -> "CountVectorizer":
    """Construit un vectoriseur Bag-of-Words non entraîné.

    Le vectoriseur transforme une liste de textes nettoyés en une matrice
    creuse de comptes de mots. On limite le vocabulaire à `max_features`
    termes pour garder un modèle léger et éviter le sur-apprentissage.

    Args:
        max_features: taille maximale du vocabulaire retenu.

    Returns:
        Une instance de CountVectorizer prête à être `fit`.
    """
    # Import local : évite de charger scikit-learn tant que ce n'est pas utile.
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(stop_words=FRENCH_STOPWORDS, max_features=max_features)


def clean_corpus(texts) -> list[str]:
    """Applique `clean_text` à une collection de textes."""
    return [clean_text(t) for t in texts]

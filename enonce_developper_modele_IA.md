

Détection de discours haineux avec une régression
logistique
Vous travaillez pour ModeraTech, une entreprise spécialisée dans la création de solutions d’intelligence artificielle
pour aider les plateformes numériques à modérer leurs contenus.
L’un de vos clients, VidéoPartage, est une grande plateforme française de partage de vidéos. Chaque jour, des
milliers de commentaires sont publiés sous les vidéos : avis, débats, encouragements, critiques, mais aussi parfois des
messages agressifs ou haineux.
Depuis quelques mois, l’équipe de modération de VidéoPartage est débordée. Les modérateurs doivent traiter un
volume de commentaires trop important, et certains messages problématiques restent visibles trop longtemps. Cela
nuit à l’expérience des utilisateurs, expose les créateurs à des attaques répétées et peut dégrader l’image de la
plateforme.
À la suite d’un incident médiatisé survenu après la publication d’une vidéo très populaire, la direction de
VidéoPartage demande à ModeraTech de proposer un premier prototype capable de détecter automatiquement les
commentaires potentiellement haineux, comme le ferait son équipe (mimétisation).
Votre mission est de construire une première version de ce système.
Situation professionnelle
Vous intégrez une petite équipe projet composée :
d’un chef de projet, chargé de cadrer le besoin client ;
d’un data analyst, chargé de préparer les données ;
d’un développeur IA, chargé d’entraîner le modèle ;
d’un responsable modération, chargé d’interpréter les résultats.
Le client ne vous demande pas encore un outil parfait. Il souhaite d’abord un prototype fonctionnel permettant de
vérifier si une approche simple de Machine Learning peut aider à repérer automatiquement certains commentaires
problématiques.
Le système devra analyser un commentaire écrit en français et le classer dans l’une des deux catégories suivantes :
haineux ;
non haineux.
Problématique
La modération manuelle est coûteuse, lente et difficile à maintenir lorsque le volume de commentaires augmente.
VidéoPartage souhaite donc savoir si un modèle simple peut aider à répondre à la question suivante :
À partir du texte d’un commentaire, peut-on prédire automatiquement s’il doit être signalé comme haineux ou non
haineux ?
Étape 0 : Préparation de l’environnement
Avant de commencer, vous devez disposer de Python et installer les bibliothèques nécessaires.
Vous aurez besoin des packages suivants :


pandas pour manipuler les données ;
scikit-learn pour vectoriser les textes, entraîner le modèle et évaluer ses performances.
Installation :
pip install pandas scikit-learn
Étape 1 : Création du jeu de données
L’équipe de ModeraTech vous fournit un petit jeu de données synthétique.
Chaque ligne contient :
un commentaire ;
un label.
Le label vaut :
1 = commentaire haineux
0 = commentaire non haineux
Attention : ce jeu de données est volontairement petit. Il sert à comprendre la démarche, pas à créer un vrai système
de modération utilisable en production.
import pandas as pd
# Dataset synthétique
data = {
    "text": [
        "Je te déteste, tu es horrible !",  # Haineux
        "J'aime beaucoup cette vidéo, merci.",  # Non haineux
        "Va te faire voir, imbécile.",  # Haineux
        "Quel contenu inspirant, bravo à l'équipe !",  # Non haineux
        "Tu es vraiment nul et inutile.",  # Haineux
        "Je suis impressionné par la qualité de cette vidéo.",  # Non haineux
        "Ferme-la, personne ne veut entendre ça.",  # Haineux
        "C'est une discussion constructive, merci pour vos efforts.",  # Non haineux
        "Ce commentaire est complètement stupide et inutile.",  # Haineux
        "Merci pour cette vidéo, elle m'a beaucoup aidé !",  # Non haineux
        "Personne n'a besoin de voir des bêtises pareilles.",  # Haineux
        "Excellent contenu, continuez comme ça !",  # Non haineux
        "Tu ne comprends rien, arrête de commenter.",  # Haineux
        "Bravo, c'est exactement ce que je cherchais.",  # Non haineux
        "Espèce d'idiot, tu ne sais même pas de quoi tu parles.",  # Haineux
        "Cette vidéo est très claire, merci pour le travail.",  # Non haineux
        "Tu es une honte, personne ne veut lire ça.",  # Haineux
        "Le tutoriel est super bien expliqué, merci !",  # Non haineux
        "C'est complètement débile, arrête de poster.",  # Haineux
        "J'adore cette chaîne, toujours des vidéos intéressantes.",  # Non haineux
        "Dégage d'ici, personne ne te supporte.",  # Haineux
        "Merci pour ces conseils, c'est vraiment utile.",  # Non haineux


        "T'es vraiment le pire, tes vidéos sont nulles.",  # Haineux
        "Une très bonne vidéo, claire et précise, bravo !"  # Non haineux
    ],
    "label": [
        1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0
    ]
}
# Conversion en DataFrame
df = pd.DataFrame(data)
print(df)
Étape 2 : Prétraitement des données
Les modèles de Machine Learning ne comprennent pas directement le langage naturel. Avant d’entraîner le modèle, il
faut transformer les commentaires en une forme exploitable.
Vous allez donc :
1. mettre les textes en minuscules ;
2. supprimer les caractères spéciaux ;
3. retirer certains mots fréquents peu informatifs ;
4. transformer les textes en vecteurs numériques.
Cette étape est importante : un mauvais prétraitement peut dégrader les performances du modèle.
import re
from sklearn.feature_extraction.text import CountVectorizer
# Fonction de nettoyage
def clean_text(text):
    text = text.lower()  # Mettre en minuscule
    text = re.sub(r"[^\w\s]", "", text)  # Supprimer les caractères spéciaux
    return text
# Appliquer le nettoyage
df["text_clean"] = df["text"].apply(clean_text)
# Liste simple de mots vides en français
french_stopwords = [
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "et", "en", "au",
    "aux", "avec", "ce", "ces", "pour", "par", "sur", "pas", "plus", "où", "mais",
    "ou", "donc", "ni", "car", "ne", "que", "qui", "quoi", "quand", "à", "son",
    "sa", "ses", "ils", "elles", "nous", "vous", "est", "sont", "cette", "cet",


    "aussi", "être", "avoir", "faire", "comme", "tout", "bien", "mal", "on", "lui"
]
# Vectorisation avec Bag of Words
vectorizer = CountVectorizer(
    stop_words=french_stopwords,
    max_features=100
)
X = vectorizer.fit_transform(df["text_clean"])
y = df["label"]
print("Vectorisation terminée.")
Étape 3 : Entraînement du modèle
L’équipe projet choisit une première approche simple : la régression logistique.
Même si son nom contient le mot régression, la régression logistique est souvent utilisée pour des problèmes de
classification.
Dans ce TP, elle va apprendre à distinguer deux classes :
0 = non haineux
1 = haineux
Vous devez diviser les données en deux parties :
un ensemble d’entraînement pour apprendre ;
un ensemble de test pour évaluer le modèle sur des données qu’il n’a pas vues pendant l’entraînement.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
# Entraînement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

---


Étape 4 : Évaluation des performances
Une fois le modèle entraîné, il faut vérifier s’il se comporte correctement.
Pour cela, vous allez comparer :
les vraies réponses attendues
les prédictions du modèle
Vous afficherez :
un rapport de classification
une matrice de confusion
Ces résultats doivent ensuite être interprétés. Un modèle peut sembler correct sur quelques exemples, mais rester
fragile si les données sont trop peu nombreuses ou trop simples.
from sklearn.metrics import classification_report, confusion_matrix
# Prédictions
y_pred = model.predict(X_test)
# Rapport de classification
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
# Matrice de confusion
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
Étape 5 : Test sur de nouveaux commentaires
Le client souhaite voir comment le prototype réagit face à de nouveaux commentaires.
Vous allez donc tester le modèle sur des phrases qui ne faisaient pas partie du jeu de données initial.
# Nouvelles données
new_comments = [
    "Je ne supporte pas cette personne.",  # Haineux
    "Cette vidéo est incroyable, merci pour votre travail.",  # Non haineux
    "Arrête de dire n'importe quoi, imbécile.",  # Haineux
    "Une excellente présentation, bravo à toute l'équipe."  # Non haineux
]
# Nettoyage et vectorisation
new_comments_clean = [
    clean_text(comment)
    for comment in new_comments
]

---

new_comments_vectorized = vectorizer.transform(new_comments_clean)
# Prédictions
predictions = model.predict(new_comments_vectorized)
for comment, label in zip(new_comments, predictions):
    prediction_label = "Haineux" if label == 1 else "Non haineux"
    print(f"Commentaire : '{comment}' -> {prediction_label}")

# TP Final — API d'analyse de sentiments (SocialMetrics AI)

API REST **Flask** qui évalue le sentiment de tweets en français à l'aide de
**deux modèles de régression logistique** (scikit-learn), avec stockage des
données annotées en **MySQL** et réentraînement automatisé.

## Pourquoi deux modèles ?

Un tweet peut être **positif**, **négatif** ou **neutre**. Un seul classifieur
binaire ne peut pas représenter le cas neutre. On entraîne donc :

- `model_positive` → probabilité que le tweet soit positif ;
- `model_negative` → probabilité que le tweet soit négatif.

Le score final, entre **-1** (très négatif) et **+1** (très positif), est :

```
score = P(positif) − P(négatif)
```

Un score proche de 0 correspond à un tweet neutre.

## Architecture

```
config.py              Configuration (env, chemins, MySQL, Flask)
data/tweets_seed.py    Jeu de données annoté d'amorçage
scripts/
  schema.sql           Création base + table (SQL pur)
  setup_database.py    Création base + table + seed (Python)
  retrain.py           Réentraînement hebdomadaire
  crontab.txt          Planification (cron / Tâches Windows)
src/
  preprocessing.py     Nettoyage + vectorisation Bag-of-Words
  database.py          Accès MySQL (requêtes paramétrées)
  train.py             Entraînement des 2 modèles
  model.py             Chargement + calcul du score
  app.py               API Flask
  evaluate.py          Matrices de confusion + rapport PDF
```

## Prérequis : un serveur MySQL

L'API se connecte à un serveur MySQL sur `localhost:3306`. Deux options :

### Option A — MySQL via Docker (recommandée, rien à installer sur la machine)

```bash
# Démarrer un serveur MySQL 8.0 dans un conteneur
docker run --name socialmetrics-mysql \
  -e MYSQL_ROOT_PASSWORD=changeme \
  -p 3306:3306 -d mysql:8.0

# Vérifier qu'il est prêt (doit afficher « mysqld is alive »)
docker exec socialmetrics-mysql mysqladmin ping -uroot -pchangeme
```

Le fichier `.env` doit alors contenir `DB_USER=root` et `DB_PASSWORD=changeme`.

### Option B — MySQL installé localement

Installer MySQL (winget, MySQL Installer, XAMPP…) puis adapter `DB_USER` /
`DB_PASSWORD` dans `.env` selon votre configuration.

> Le client en ligne de commande `mysql` n'est **pas** nécessaire : l'accès se
> fait uniquement via `mysql-connector-python`.

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis renseigner les identifiants MySQL
```

## Mise en route

```bash
# 0. Démarrer MySQL (voir « Prérequis » ci-dessus)

# 1. Créer la base MySQL, la table `tweets` et insérer les données d'amorçage
python scripts/setup_database.py

# 2. Entraîner les deux modèles
python src/train.py

# 3. Lancer l'API (à laisser tourner dans un terminal dédié)
python src/app.py
```

L'API est alors disponible sur `http://localhost:5000`.

## Commandes utiles

```bash
# Relancer l'API plus tard
source .venv/Scripts/activate && python src/app.py

# Générer le rapport d'évaluation (PNG + PDF dans reports/)
python src/evaluate.py

# Gérer le conteneur MySQL (données conservées entre les redémarrages)
docker stop socialmetrics-mysql     # arrêter
docker start socialmetrics-mysql    # redémarrer
docker rm -f socialmetrics-mysql    # supprimer complètement
```

## Utilisation de l'API

### `POST /analyze`

Requête :

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweets": ["J\'adore ce produit", "Service catastrophique"]}'
```

Réponse :

```json
{
  "J'adore ce produit": 0.82,
  "Service catastrophique": -0.79
}
```

### `GET /health`

Retourne `{"status": "ok"}` pour la supervision.

### Gestion des erreurs

| Cas                          | Code HTTP |
| ---------------------------- | --------- |
| Corps JSON absent            | 400       |
| Champ `tweets` manquant      | 400       |
| `tweets` n'est pas une liste | 400       |
| Liste vide                   | 400       |
| Élément non textuel          | 400       |
| Modèles non entraînés        | 503       |

## Réentraînement automatisé

Le script `scripts/retrain.py` relance l'entraînement sur les données les plus
récentes. Voir `scripts/crontab.txt` pour la planification hebdomadaire
(cron sous Linux/macOS, Planificateur de tâches sous Windows). Les journaux
sont écrits dans `logs/retrain.log`.

## Rapport d'évaluation

```bash
python src/evaluate.py
```

Génère les deux matrices de confusion (`reports/confusion_positive.png`,
`reports/confusion_negative.png`) et le rapport PDF
`reports/rapport_evaluation.pdf` (précision, rappel, F1-score par classe).

## Base de données

Table `tweets` :

| Colonne  | Type         | Description           |
| -------- | ------------ | --------------------- |
| id       | INT (PK)     | Identifiant unique    |
| text     | VARCHAR(512) | Contenu du tweet      |
| positive | TINYINT(1)   | 1 si positif, 0 sinon |
| negative | TINYINT(1)   | 1 si négatif, 0 sinon |

Toutes les requêtes utilisent des **paramètres liés** (`%s`) pour prévenir les
injections SQL.

"""Couche d'accès à la base de données MySQL.

Toutes les requêtes utilisent des paramètres liés (placeholders %s) afin
d'éviter toute injection SQL. Le schéma correspond à la table `tweets`
demandée dans l'énoncé : id, text, positive, negative.
"""
from __future__ import annotations

from typing import Iterable

import mysql.connector
import pandas as pd

from config import DB_CONFIG, DB_NAME

# --- Schéma de la table -------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tweets (
    id       INT AUTO_INCREMENT PRIMARY KEY,
    text     VARCHAR(512) NOT NULL,
    positive TINYINT(1) NOT NULL DEFAULT 0,
    negative TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def get_server_connection():
    """Connexion au serveur MySQL sans sélectionner de base (pour la créer)."""
    return mysql.connector.connect(**DB_CONFIG)


def get_connection():
    """Connexion à la base `socialmetrics` (celle qui contient la table)."""
    return mysql.connector.connect(database=DB_NAME, **DB_CONFIG)


def init_database() -> None:
    """Crée la base de données puis la table `tweets` si elles n'existent pas."""
    conn = get_server_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def count_tweets() -> int:
    """Retourne le nombre de tweets stockés."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tweets")
        (total,) = cursor.fetchone()
        cursor.close()
        return int(total)
    finally:
        conn.close()


def insert_tweets(rows: Iterable[tuple[str, int, int]]) -> int:
    """Insère un lot de tweets annotés.

    Args:
        rows: itérable de tuples (text, positive, negative).

    Returns:
        Le nombre de lignes insérées.
    """
    rows = list(rows)
    if not rows:
        return 0
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # executemany = insertion par lot (bien plus efficace que N requêtes).
        cursor.executemany(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            rows,
        )
        conn.commit()
        inserted = cursor.rowcount
        cursor.close()
        return inserted
    finally:
        conn.close()


def fetch_tweets() -> pd.DataFrame:
    """Charge tous les tweets annotés dans un DataFrame pandas."""
    conn = get_connection()
    try:
        return pd.read_sql(
            "SELECT id, text, positive, negative FROM tweets ORDER BY id",
            conn,
        )
    finally:
        conn.close()

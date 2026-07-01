-- Script SQL de création de la base et de la table `tweets`.
-- Alternative pure SQL au script Python scripts/setup_database.py.
-- Usage : mysql -u root -p < scripts/schema.sql

CREATE DATABASE IF NOT EXISTS socialmetrics
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE socialmetrics;

CREATE TABLE IF NOT EXISTS tweets (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    text       VARCHAR(512) NOT NULL,
    positive   TINYINT(1) NOT NULL DEFAULT 0,
    negative   TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

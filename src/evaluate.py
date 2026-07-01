"""Génération du rapport d'évaluation (matrices de confusion + PDF).

Produit, pour les deux modèles (positif et négatif) :
  - une image PNG de la matrice de confusion ;
  - les métriques précision / rappel / F1 par classe ;
puis assemble le tout dans un rapport PDF (reports/rapport_evaluation.pdf).

Usage :
    python src/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # backend sans interface graphique (génération de fichiers).
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import REPORTS_DIR
from src.database import fetch_tweets
from src.preprocessing import build_vectorizer, clean_corpus
from sklearn.linear_model import LogisticRegression


def _plot_confusion(y_true, y_pred, title: str, filename: str) -> Path:
    """Sauvegarde la matrice de confusion en PNG et renvoie son chemin."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["classe 0", "classe 1"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    path = REPORTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _evaluate_model(name, y_train, y_test, X_train, X_test, png_name):
    """Entraîne un modèle, calcule ses métriques et sa matrice de confusion."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    png_path = _plot_confusion(
        y_test, y_pred, f"Matrice de confusion — {name}", png_name
    )
    return report, png_path


def _build_pdf(pos_report, pos_png, neg_report, neg_png) -> Path:
    """Assemble un PDF récapitulatif à partir des rapports et des images."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Rapport d'évaluation - Analyse de sentiments", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Deux modèles de régression logistique sont évalués : un modèle "
        "'positif' et un modèle 'négatif'. Les matrices de confusion et les "
        "métriques (précision, rappel, F1-score) sont présentées ci-dessous.",
    )

    for title, report, png in (
        ("Modèle positif", pos_report, pos_png),
        ("Modèle négatif", neg_report, neg_png),
    ):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Courier", "", 9)
        for line in report.splitlines():
            pdf.cell(0, 5, line, ln=True)
        pdf.image(str(png), w=90)

    out = REPORTS_DIR / "rapport_evaluation.pdf"
    pdf.output(str(out))
    return out


def main() -> None:
    df = fetch_tweets()
    if df.empty:
        raise SystemExit("Table `tweets` vide : lancez scripts/setup_database.py.")

    texts = clean_corpus(df["text"].tolist())
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    y_pos = df["positive"].to_numpy()
    y_neg = df["negative"].to_numpy()

    idx = list(range(X.shape[0]))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.25, random_state=42, stratify=y_pos
    )
    X_train, X_test = X[idx_train], X[idx_test]

    pos_report, pos_png = _evaluate_model(
        "positif", y_pos[idx_train], y_pos[idx_test], X_train, X_test,
        "confusion_positive.png",
    )
    neg_report, neg_png = _evaluate_model(
        "négatif", y_neg[idx_train], y_neg[idx_test], X_train, X_test,
        "confusion_negative.png",
    )

    print("=== Modèle positif ===")
    print(pos_report)
    print("=== Modèle négatif ===")
    print(neg_report)

    pdf_path = _build_pdf(pos_report, pos_png, neg_report, neg_png)
    print(f"Rapport PDF généré : {pdf_path}")


if __name__ == "__main__":
    main()

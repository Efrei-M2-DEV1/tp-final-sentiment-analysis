"""Jeu de données annoté de tweets en français.

Chaque entrée est un tuple : (text, positive, negative)
  - positive = 1 si le tweet exprime un sentiment positif, sinon 0
  - negative = 1 si le tweet exprime un sentiment négatif, sinon 0

Un tweet neutre a positive=0 ET negative=0. C'est précisément parce qu'un
tweet peut être positif, négatif ou neutre que l'on entraîne DEUX modèles
binaires distincts plutôt qu'un seul.

Ce dataset synthétique sert à amorcer la table MySQL `tweets`. Il est
volontairement de taille modeste (démonstration du prototype).
"""

# (texte, positive, negative)
SEED_TWEETS = [
    # --- Tweets positifs ---------------------------------------------------
    ("J'adore ce nouveau produit, il est vraiment génial !", 1, 0),
    ("Merci beaucoup pour ce service impeccable, bravo à l'équipe.", 1, 0),
    ("Quelle superbe journée, je suis de très bonne humeur.", 1, 0),
    ("Ce film était incroyable, un vrai chef-d'oeuvre.", 1, 0),
    ("Félicitations pour cette réussite bien méritée !", 1, 0),
    ("Le concert d'hier soir était magique, un moment inoubliable.", 1, 0),
    ("Super livraison, rapide et bien emballée, je recommande.", 1, 0),
    ("Je suis tellement fier de mon équipe aujourd'hui.", 1, 0),
    ("Excellente initiative, ça fait vraiment plaisir à voir.", 1, 0),
    ("Un grand merci, votre aide a tout changé pour moi.", 1, 0),
    ("Cette mise à jour est fantastique, l'appli est bien plus fluide.", 1, 0),
    ("Que du bonheur avec ce restaurant, cuisine délicieuse !", 1, 0),
    ("Bravo pour cette performance, c'était impressionnant.", 1, 0),
    ("J'ai passé un moment formidable, merci pour tout.", 1, 0),
    ("Le support client a été adorable et très efficace.", 1, 0),
    ("Magnifique paysage, ce voyage restera gravé dans ma mémoire.", 1, 0),
    ("Content d'avoir choisi cette formation, elle est excellente.", 1, 0),
    ("Trop heureux, mon équipe a gagné le match ce soir !", 1, 0),
    ("Produit au top, rapport qualité prix imbattable.", 1, 0),
    ("Merci pour ce tutoriel clair et bien expliqué, parfait.", 1, 0),

    # --- Tweets négatifs ---------------------------------------------------
    ("Je déteste ce service, une perte de temps totale.", 0, 1),
    ("Quelle catastrophe, le produit est arrivé cassé.", 0, 1),
    ("Franchement nul, je ne recommande à personne.", 0, 1),
    ("Le film était ennuyeux et beaucoup trop long.", 0, 1),
    ("Encore un retard, ce transporteur est une honte.", 0, 1),
    ("Très déçu de la qualité, c'est de l'arnaque.", 0, 1),
    ("Le support client est incompétent et désagréable.", 0, 1),
    ("Application horrible, elle plante sans arrêt.", 0, 1),
    ("Je suis furieux, ma commande n'est jamais arrivée.", 0, 1),
    ("Un vrai gâchis, plus jamais je n'achèterai ici.", 0, 1),
    ("Cette mise à jour a tout cassé, c'est inadmissible.", 0, 1),
    ("Repas immangeable, service lent, une soirée gâchée.", 0, 1),
    ("Ras-le-bol de ces bugs, l'appli est inutilisable.", 0, 1),
    ("Prix scandaleux pour une qualité aussi médiocre.", 0, 1),
    ("Je regrette cet achat, quelle grosse déception.", 0, 1),
    ("Ambiance exécrable, le personnel était odieux.", 0, 1),
    ("Le pire concert de ma vie, une vraie déception.", 0, 1),
    ("Site incompréhensible, impossible de finaliser mon panier.", 0, 1),
    ("Tellement en colère, on m'a facturé deux fois.", 0, 1),
    ("Formation médiocre, j'ai perdu mon argent.", 0, 1),

    # --- Tweets neutres (ni positif, ni négatif) ---------------------------
    ("La réunion est prévue à quatorze heures demain.", 0, 0),
    ("Il pleut sur Paris cet après-midi.", 0, 0),
    ("Le colis sera livré entre lundi et mercredi.", 0, 0),
    ("Voici le lien vers la documentation officielle.", 0, 0),
    ("La conférence commence à neuf heures dans la salle B.", 0, 0),
    ("Le magasin ferme à vingt heures ce soir.", 0, 0),
    ("J'ai mis à jour le fichier partagé ce matin.", 0, 0),
    ("Le train part du quai numéro cinq.", 0, 0),
    ("La météo annonce des nuages pour le week-end.", 0, 0),
    ("Le rendez-vous a été déplacé à jeudi prochain.", 0, 0),
]

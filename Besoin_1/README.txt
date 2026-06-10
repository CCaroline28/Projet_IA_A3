Exploration et cartographie des bornes de recharge
===================================================

Ce script génère des graphiques et des cartes interactives à partir
des données de bornes de recharge. Tout est sauvegardé en local,
rien ne s'affiche directement à l'écran.


Avant de commencer
------------------

Avoir le fichier export_IA.csv dans le même dossier, puis installe
les dépendances :

  pip install pandas folium matplotlib seaborn


Comment lancer
--------------

  python exploration_irve.py


Ce que ça produit
-----------------

Graphiques (PNG) :
  - distribution_puissance_nominale.png  : histogramme des puissances
  - type_implantation.png                : répartition voirie / parking / etc.
  - condition_acces.png                  : accès libre, réservé, etc.
  - types_prises.png                     : EF, Type 2, CCS, CHAdeMO

Cartes interactives (HTML, à ouvrir dans un navigateur) :
  - carte_implantation.html  : localisation des bornes par type d'implantation
  - carte_puissance.html     : localisation par catégorie de puissance
  - heatmap_irve.html        : densité des bornes sur le territoire


Script d'exemple
----------------

Ce bout de code recharge les données et prédit la catégorie de puissance
d'une borne, comme le fait le script en interne :

  import pandas as pd

  def categorie_puissance(p):
      if p <= 22:
          return "Normale"
      elif p <= 50:
          return "Rapide"
      elif p <= 150:
          return "Très rapide"
      else:
          return "Ultra rapide"

  df = pd.read_csv("export_IA.csv")
  df["categorie_puissance"] = df["puissance_nominale"].apply(categorie_puissance)

  print(df["categorie_puissance"].value_counts())
  # → Normale        XXXX
  #   Rapide          XXX
  #   Très rapide     XXX
  #   Ultra rapide     XX

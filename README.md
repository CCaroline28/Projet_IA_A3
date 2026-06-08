### Justification du choix des variables sélectionnées

Pour répondre au besoin de visualisation spatiale des infrastructures de recharge pour véhicules électriques (IRVE), plusieurs variables ont été retenues :

* **consolidated_latitude** et **consolidated_longitude** : ces variables permettent de localiser géographiquement chaque borne de recharge sur le territoire français. Elles sont indispensables pour la création des cartes interactives et de la carte de chaleur.

* **implantation_station** : cette variable décrit le type d’implantation de la station (parking public, voirie, commerce, etc.). Elle permet d’étudier la répartition géographique des bornes selon leur environnement d’installation.

* **puissance_nominale** : cette variable renseigne la puissance maximale délivrée par la borne de recharge. Elle permet de distinguer les bornes normales, rapides, très rapides et ultra-rapides afin d’analyser leur distribution sur le territoire.

Ces variables ont été sélectionnées car elles répondent directement aux objectifs du besoin client : visualiser la répartition spatiale des bornes selon leur implantation et leur puissance, ainsi que leur densité géographique.

---

### Justification des opérations de préparation des données

Plusieurs opérations de préparation ont été réalisées afin de garantir la qualité des visualisations produites.

1. **Sélection des variables utiles**

   Seules les variables nécessaires à la création des cartes ont été conservées afin de réduire la taille du jeu de données et d'améliorer les performances de traitement.

2. **Suppression des valeurs manquantes**

   Les lignes contenant des coordonnées géographiques manquantes ont été supprimées car elles ne peuvent pas être représentées sur une carte.

3. **Conversion des coordonnées en format numérique**

   Les variables de latitude et de longitude ont été converties au format numérique afin de permettre leur utilisation par les bibliothèques de visualisation géographique.

4. **Filtrage géographique**

   Un filtrage a été appliqué afin de conserver uniquement les bornes situées en France métropolitaine. Cette opération permet d’éliminer d’éventuelles coordonnées erronées ou situées hors de la zone d’étude.

5. **Création de catégories de puissance**

   Les puissances nominales ont été regroupées en quatre catégories :

   * Normale ≤ 22 kW
   * Rapide 23-50 kW
   * Très rapide 51-150 kW
   * Ultra rapide > 150 kW

   Cette catégorisation facilite l’interprétation visuelle et permet une comparaison plus simple des différents niveaux de puissance.

6. **Agrégation spatiale pour la heatmap**

   Les coordonnées ont été regroupées par zones géographiques proches afin de mettre davantage en évidence les concentrations de bornes et d’éviter qu’une seule grande agglomération ne masque les autres zones d’intérêt.

---

### Justification des méthodes de création des cartes

Trois types de représentations cartographiques ont été réalisés.

#### Carte selon le type d’implantation

Une carte à points colorés a été utilisée afin de représenter chaque borne selon son type d’implantation. Cette représentation permet d’identifier rapidement les zones où certains types d’implantation sont plus fréquents.

#### Carte selon la puissance nominale

Une seconde carte à points colorés a été construite afin de représenter les différentes catégories de puissance. Les couleurs permettent de distinguer visuellement les bornes normales, rapides, très rapides et ultra-rapides et d’analyser leur répartition géographique.

#### Carte de chaleur (Heatmap)

Une carte de chaleur a été utilisée pour représenter la densité spatiale des bornes de recharge. Cette méthode est particulièrement adaptée à l’identification des zones de forte concentration. Les couleurs chaudes (jaune, orange, rouge) mettent en évidence les territoires les plus équipés tandis que les couleurs froides (bleu, cyan) correspondent aux zones moins denses.

Cette représentation permet d’identifier rapidement les principaux pôles de déploiement des infrastructures de recharge en France, notamment autour des grandes métropoles et des axes de mobilité importants.

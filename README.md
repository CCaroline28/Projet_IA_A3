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

########################################
Methode de coude 
Analyse complète des métriques
1. Coefficient de Silhouette

Rappel :

proche de 1 → excellent clustering
proche de 0 → clusters qui se chevauchent
négatif → mauvais clustering

Résultats :

K	Silhouette
4	0.4729
5	0.5028
6	0.5013
12	0.4977

👉 Le meilleur score est obtenu pour :

K = 5

C'est un argument très fort.

2. Indice de Calinski-Harabasz

Rappel :

Plus la valeur est élevée, mieux les clusters sont :
compacts à l'intérieur
séparés entre eux

Résultats :

K	Calinski
5	55424
10	58646
12	63581
15	64353

On constate que :

Le score augmente presque continuellement
lorsque K augmente.

C'est un comportement classique.

Donc si on regardait uniquement Calinski, on choisirait :

K = 15

Mais cela produirait beaucoup de clusters, difficiles à interpréter géographiquement.

👉 C'est pourquoi Calinski ne doit pas être utilisé seul.

3. Indice de Davies-Bouldin

Rappel :

Plus petit = meilleur

Résultats :

K	Davies
5	0.6905
11	0.6921
12	0.6785

Le meilleur score est :

K = 12

Mais l'amélioration par rapport à K=5 est faible :

0.6905 → 0.6785

Alors qu'on passe de :

5 clusters → 12 clusters

La perte en lisibilité n'est donc pas justifiée.

Analyse globale correcte
Critère	Meilleur K
Méthode du coude	5
Silhouette	5
Calinski-Harabasz	15
Davies-Bouldin	12

On observe que :

la méthode du coude indique K≈5 ;
le coefficient de silhouette est maximal pour K=5 ;
Calinski-Harabasz favorise des valeurs élevées de K ;
Davies-Bouldin est minimal pour K=12 mais l'amélioration reste faible.

Ainsi :

K = 5

constitue le meilleur compromis entre :

qualité du clustering,
interprétation géographique,
simplicité de visualisation.
Formulation rapport

Le choix du nombre de clusters a été réalisé à l'aide de la méthode du coude et de trois métriques d'évaluation. La méthode du coude montre une rupture de pente autour de K=5. Le coefficient de silhouette atteint sa valeur maximale pour K=5 (0,5028), indiquant une bonne séparation des groupes. Bien que l'indice de Calinski-Harabasz continue d'augmenter pour des valeurs de K plus élevées et que l'indice de Davies-Bouldin soit légèrement meilleur pour K=12, ces solutions produisent un découpage plus fin et moins interprétable. Le choix final s'est donc porté sur K=5, qui représente le meilleur compromis entre performance et lisibilité des clusters.

##################CARTE #####################
Répartition des bornes par cluster :
cluster
0    14478
1    10690
2     7497
3     4971
4     7085
Name: count, dtype: int64
########### Script + Exel ##################
### Script de prédiction du cluster

Afin de répondre à la dernière exigence du besoin 2, un script de prédiction a été développé. Ce script permet de déterminer automatiquement le cluster associé à une borne de recharge à partir de ses coordonnées géographiques (latitude et longitude).

Le modèle K-Means n'est pas réentraîné à chaque utilisation. Conformément aux consignes du projet, le modèle préalablement entraîné est sauvegardé puis chargé directement lors de l'exécution du script. Cette approche permet de réduire considérablement le temps de calcul et garantit la cohérence des résultats obtenus.

L'utilisateur saisit les coordonnées de la borne à analyser, puis le script charge le modèle enregistré et renvoie le cluster correspondant.

Afin de faciliter l'utilisation du script, un fichier Excel complémentaire contenant les coordonnées moyennes des principales communes a également été généré. Ce fichier permet à un utilisateur ne connaissant pas les coordonnées GPS exactes d'une borne de retrouver facilement les coordonnées associées à une ville donnée avant de lancer la prédiction.

Cette solution améliore l'ergonomie du système tout en conservant la logique du modèle de clustering, qui repose exclusivement sur les coordonnées géographiques utilisées lors de l'apprentissage.

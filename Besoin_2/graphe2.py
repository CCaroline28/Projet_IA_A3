import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Résultats déjà calculés

K = list(range(2, 16))

silhouettes = [
    0.4217, 0.4567, 0.4729, 0.5028,
    0.5013, 0.5005, 0.4625, 0.4676,
    0.4856, 0.4878, 0.4977, 0.4882,
    0.4925, 0.4900
]

calinski_scores = [
    34400.56, 41062.19, 46261.88, 55424.38,
    55047.47, 55731.26, 56680.08, 57810.32,
    58646.54, 60894.30, 63581.32, 63117.61,
    63171.01, 64353.10
]

davies_scores = [
    1.0153, 0.7675, 0.7239, 0.6905,
    0.7520, 0.7560, 0.7785, 0.7636,
    0.7143, 0.6921, 0.6785, 0.6982,
    0.7191, 0.7031
]

scores = pd.DataFrame({
    "K": K,
    "Silhouette": silhouettes,
    "Calinski": calinski_scores,
    "Davies": davies_scores
})

scaler = MinMaxScaler()

scores_norm = scores.copy()

scores_norm[
    ["Silhouette", "Calinski", "Davies"]
] = scaler.fit_transform(
    scores[
        ["Silhouette", "Calinski", "Davies"]
    ]
)

# inversion Davies car plus petit = meilleur
scores_norm["Davies"] = 1 - scores_norm["Davies"]

plt.figure(figsize=(10,6))

plt.plot(
    scores_norm["K"],
    scores_norm["Silhouette"],
    marker="o",
    linewidth=2,
    label="Silhouette"
)

plt.plot(
    scores_norm["K"],
    scores_norm["Calinski"],
    marker="s",
    linewidth=2,
    label="Calinski-Harabasz"
)

plt.plot(
    scores_norm["K"],
    scores_norm["Davies"],
    marker="^",
    linewidth=2,
    label="Davies-Bouldin"
)

plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Score normalisé")
plt.title("Comparaison des métriques de clustering")

plt.legend()
plt.grid(True)

plt.savefig(
    "comparaison_metriques_clustering.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
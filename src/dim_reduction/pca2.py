from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
csv_path = Path("../sc2egset.csv").resolve().as_posix()
df = pd.read_csv(csv_path)

from sklearn.preprocessing import StandardScaler

X = df[["gameloop", "game_hash", "mineralsCollectionRate", "workersActiveCount"]]

scaller = StandardScaler()
pca = PCA(random_state=42)

components = pca.fit_transform(X)
components = scaller.fit_transform(components)

labels = {
    str(i): f"PCA {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(4),
    color=df["workersActiveCount"]
)
fig.update_traces(diagonal_visible=False)
fig.show()
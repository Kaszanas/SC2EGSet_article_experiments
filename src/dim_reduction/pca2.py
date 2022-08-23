from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.dim_reduction.utils.replacement import Replacement

csv_path = Path("../sc2egset.csv").resolve().as_posix()
df = pd.read_csv(csv_path)

Replacement.race_name_into_number_value(df)
Replacement.outcome_into_number_value(df)
Replacement.map_into_number_value(df)
Replacement.player_name_into_number_value(df)
Replacement.player_toon_into_number_value(df)

features = df.columns
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=3, random_state=42)
components = pca.fit_transform(x)

labels = {
    str(i): f"PC {i + 1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(3),
    color=df["race"]
)
fig.update_traces(diagonal_visible=False)
fig.show()

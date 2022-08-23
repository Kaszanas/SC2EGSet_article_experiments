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
# Separating out the target
y = df.loc[:, ['race']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(x)

fig = px.scatter(
    components,
    x=0,
    y=1,
    color=df['race']
)
fig.show()

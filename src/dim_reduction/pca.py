from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# read csv file
csv_path = Path("../sc2egset.csv").resolve().as_posix()
df = pd.read_csv(csv_path)

# all csv columns, with excluded: 'map_name', 'player_name', 'player_toon', 'race', 'outcome'
x = df.loc[:, ~df.columns.isin(['map_name', 'player_name', 'player_toon', 'race', 'outcome'])]
# standardizing the features
x = StandardScaler().fit_transform(x)

# changing csv incomplete race name to correct in game races names
df['race'].replace({"Terr": "Terran", "Prot": "Protoss"}, inplace=True)

# pca
# n_components = 5 cuz my pc is terrible
n_components = 5
pca = PCA(n_components=n_components, random_state=42)
components = pca.fit_transform(x)

# total explained variance of the data, percentage value, more n_components = better
total_var = pca.explained_variance_ratio_.sum() * 100

# labels, and title of the legend in the plot
labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
labels['color'] = "StarCraft 2 races"

# plot
fig = px.scatter_matrix(
    components,
    color=df['race'],
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()

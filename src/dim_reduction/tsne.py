# %%

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.drop_fields_util import drop_unused_fields
from utils.groupby_util import groupby_fields_mean
from utils.prepare_data_util import prep_for_dim_reduction

# %%

csv_path = Path("../sc2egset.csv").resolve().as_posix()
loaded_data = pd.read_csv(csv_path)

# %%

unique_games = loaded_data["game_hash"].nunique()
groupby_fields = ["outcome", "race", "map_name", "player_name"]
drop_fields = ["game_time_gameloop", "gameloop"]
grouped_dataframes = groupby_fields_mean(data=loaded_data, fields=groupby_fields)
grouped_dataframes = drop_unused_fields(
    grouped_dataframes=grouped_dataframes,
    fields_to_drop=drop_fields,
    unique_games=unique_games,
)

# %%

dim_reduction_models = {
    "t-SNE": TSNE(random_state=42),
    "PCA": PCA(random_state=42),
}

umap_collection = {}
for model_name, model in dim_reduction_models.items():
    pass

    for grouped_field, result_df in grouped_dataframes.items():
        standardized_data, unique_map_display = prep_for_dim_reduction(
            grouped_field=grouped_field, result_df=result_df
        )

        X, y = loaded_data['gameloop'], loaded_data['workersActiveCount']
        tsne = TSNE()
        pca = PCA(0.95)
        X_pca = pca.fit_transform(standardized_data)
        X_tsne = tsne.fit_transform(X_pca[:10000])

        matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
        proj = pd.DataFrame(X_tsne)
        proj.columns = ["comp_1", "comp_2"]
        proj["labels"] = y
        sns.lmplot("comp_1", "comp_2", hue="labels", data=proj.sample(5000), fit_reg=False)
        plt.show()

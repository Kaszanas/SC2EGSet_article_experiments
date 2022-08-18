#%%

from pathlib import Path

import pandas as pd

# UMAP:
import umap
import umap.plot

# t-SNE:
from sklearn.manifold import TSNE

# PCA
from sklearn.decomposition import PCA

from utils.prepare_data_util import prep_for_dim_reduction
from utils.drop_fields_util import drop_unused_fields


from utils.groupby_util import groupby_fields_mean

#%%

csv_path = Path("../sc2egset.csv").resolve().as_posix()
loaded_data = pd.read_csv(csv_path)

#%%

# Suggested fields might include: "map_name", "player_name"

unique_games = loaded_data["game_hash"].nunique()
groupby_fields = ["outcome", "race"]
drop_fields = ["game_time_gameloop", "gameloop"]
grouped_dataframes = groupby_fields_mean(data=loaded_data, fields=groupby_fields)
grouped_dataframes = drop_unused_fields(
    grouped_dataframes=grouped_dataframes,
    fields_to_drop=drop_fields,
    unique_games=unique_games,
)

#%%

dim_reduction_models = {
    "UMAP": umap.UMAP(random_state=42),
    "t-SNE": TSNE(random_state=42),
    "PCA": PCA(random_state=42),
}

dim_red_solved = {}
for model_name, model in dim_reduction_models.items():

    for grouped_field, result_df in grouped_dataframes.items():

        grouped_dict = {"model_name": model_name}

        standardized_data, unique_map_display = prep_for_dim_reduction(
            grouped_field=grouped_field, result_df=result_df
        )

        grouped_dict["map_display"] = unique_map_display

        reducer = model
        reducer.fit(X=standardized_data, y=result_df[grouped_field])
        print(f"UMAP for field {grouped_field}")

        grouped_dict["model"] = reducer
        dim_red_solved[f"{model_name}_{grouped_field}"] = grouped_dict


# %%


for key, obj in dim_red_solved.items():

    if obj["model_name"] == "UMAP":

        # umap.plot.output_file(filename=f"{grouped_field}_interactive_bokeh_plot.html")
        # interactive_plot = umap.plot.interactive(
        #     reducer,
        #     labels=result_df[grouped_field].map(unique_map_display),
        #     color_key_cmap="Paired",
        #     background="black",
        # )
        static_plot = umap.plot.points(
            reducer,
            labels=result_df[grouped_field].map(unique_map_display),
            color_key_cmap="Paired",
            background="black",
        )
        umap.plot.show(static_plot)
        umap.plot.output_file(filename=f"{model_name}_{grouped_field}_static_plot.html")
        # umap.plot.show(interactive_plot)

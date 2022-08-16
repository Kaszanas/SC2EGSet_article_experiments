#%%

from pathlib import Path
import umap
import umap.plot
import pandas as pd
from .utils.prepare_data_util import prep_for_dim_reduction
from utils.drop_fields_util import drop_unused_fields

from utils.groupby_util import groupby_fields_mean

#%%

csv_path = Path("../sc2egset.csv").resolve().as_posix()
loaded_data = pd.read_csv(csv_path)

#%%

unique_games = loaded_data["game_hash"].nunique()
groupby_fields = ["outcome", "race", "map_name", "player_name"]
drop_fields = ["game_time_gameloop", "gameloop"]
grouped_dataframes = groupby_fields_mean(data=loaded_data, fields=groupby_fields)
grouped_dataframes = drop_unused_fields(
    grouped_dataframes=grouped_dataframes,
    fields_to_drop=drop_fields,
    unique_games=unique_games,
)

#%%

umap_collection = {}

for grouped_field, result_df in grouped_dataframes.items():

    standardized_data, unique_map_display = prep_for_dim_reduction(
        grouped_field=grouped_field, result_df=result_df
    )

    reducer = umap.UMAP(random_state=42)
    reducer.fit(X=standardized_data, y=result_df[grouped_field])
    print(f"UMAP for field {grouped_field}")

    umap_collection[grouped_field] = reducer

    umap.plot.output_file(filename=f"{grouped_field}_interactive_bokeh_plot.html")
    interactive_plot = umap.plot.interactive(
        reducer,
        labels=result_df[grouped_field].map(unique_map_display),
        color_key_cmap="Paired",
        background="black",
    )
    static_plot = umap.plot.points(
        reducer,
        labels=result_df[grouped_field].map(unique_map_display),
        color_key_cmap="Paired",
        background="black",
    )
    umap.plot.show(static_plot)
    umap.plot.output_file(filename=f"{grouped_field}_static_bokeh_plot.html")
    umap.plot.show(interactive_plot)

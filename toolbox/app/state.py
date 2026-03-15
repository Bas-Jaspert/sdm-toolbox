from dataclasses import dataclass, field
from typing import Optional
import geopandas as gpd
import pandas as pd

@dataclass
class AppState:
    # Step 1
    species: str = ""
    country_code: str = ""
    county_name: str = ""
    year: int = 2023

    # Step 2
    data_mode: str = "explore"       # "explore" | "deepdive" | "own"
    dataset_key: str = ""
    species_gdf: Optional[gpd.GeoDataFrame] = None

    # Step 3
    selected_layers: list = field(default_factory=list)
    model_type: str = "rf"
    n_trees: int = 100
    max_depth: int = 3
    train_size: float = 0.75
    layer_stack: Optional[dict] = None   # EE images from get_layer_information()

    # Step 4
    model: object = None
    results_df: Optional[pd.DataFrame] = None
    classified_img: object = None

    # Step 5
    whatif_offsets: dict = field(default_factory=dict)

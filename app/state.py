from dataclasses import dataclass, field
from typing import Any, Optional
import geopandas as gpd
import pandas as pd


@dataclass
class AppState:
    # Step 1
    species: str = ""
    country_code: str = ""
    county_name: str = ""
    year_start: int = 2018
    year_end: int = 2025

    # Step 2
    data_mode: str = "explore"  # "explore" | "deepdive" | "own"
    dataset_key: str = ""
    gbif_user: str = ""
    gbif_pwd: str = ""
    species_gdf: Optional[gpd.GeoDataFrame] = None

    # Step 3
    selected_layers: list[str] = field(default_factory=list)
    model_type: str = "rf"
    n_trees: int = 100
    max_depth: int = 3
    train_size: float = 0.75
    resolution: int = 100
    layer_stack: Optional[dict[str, Any]] = None

    # Step 4
    model: Any = None
    results_df: Optional[pd.DataFrame] = None
    classified_img: Any = None
    ml_gdf: Optional[gpd.GeoDataFrame] = (
        None  # combined presence+background used for training
    )

    # Step 5
    whatif_offsets: dict = field(default_factory=dict)

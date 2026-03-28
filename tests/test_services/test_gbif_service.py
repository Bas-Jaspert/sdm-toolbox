"""Tests for gbif_service internal helpers."""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from app.services.gbif_service import _cleanup_gdf


def test_cleanup_gdf_preserves_year_as_int() -> None:
    """_cleanup_gdf must not coerce the year column to string."""
    gdf = gpd.GeoDataFrame(
        {"species": ["Foo bar"], "year": [2021]},
        geometry=[Point(15.0, 47.0)],
        crs="EPSG:4326",
    )
    result = _cleanup_gdf(gdf)
    assert pd.api.types.is_integer_dtype(result["year"]), (
        f"year column dtype should be integer, got {result['year'].dtype}"
    )

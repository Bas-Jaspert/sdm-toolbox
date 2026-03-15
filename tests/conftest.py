import matplotlib

matplotlib.use("Agg")

from pathlib import Path
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
from unittest.mock import MagicMock, patch

ASSETS_DIR = Path(__file__).parents[1] / "assets"


@pytest.fixture
def assets_dir():
    return ASSETS_DIR


@pytest.fixture
def background_csv_path(assets_dir):
    return assets_dir / "background_data.csv"


@pytest.fixture
def presence_csv_path(assets_dir):
    return assets_dir / "sdm_pvals_data_at_new_2.csv"


@pytest.fixture
def background_gdf(background_csv_path):
    from toolbox.utils import load_background_data

    return load_background_data(path=background_csv_path)


@pytest.fixture
def sample_presence_gdf():
    data = {
        "species": ["Lagopus muta"] * 5,
        "NDVI": [0.5, 0.6, 0.7, 0.4, 0.55],
        "elevation": [1500, 1600, 1700, 1400, 1550],
        "slope": [10, 15, 20, 8, 12],
        "aspect": [180, 200, 220, 160, 190],
        "landcover": [321, 321, 312, 321, 312],
        "snow_cover": [0.3, 0.4, 0.5, 0.2, 0.35],
        "snow_depth": [0.1, 0.15, 0.2, 0.08, 0.12],
    }
    geometries = [
        Point(11.5, 47.2),
        Point(11.6, 47.3),
        Point(11.7, 47.4),
        Point(11.4, 47.1),
        Point(11.55, 47.25),
    ]
    return gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")


@pytest.fixture
def sample_background_gdf():
    data = {
        "NDVI": [0.5, 0.6, 0.7, 0.4, 0.55],
        "elevation": [1500, 1600, 1700, 1400, 1550],
        "slope": [10, 15, 20, 8, 12],
        "aspect": [180, 200, 220, 160, 190],
        "landcover": [321, 321, 312, 321, 312],
        "snow_cover": [0.3, 0.4, 0.5, 0.2, 0.35],
        "snow_depth": [0.1, 0.15, 0.2, 0.08, 0.12],
    }
    geometries = [
        Point(12.0, 47.5),
        Point(12.1, 47.6),
        Point(12.2, 47.7),
        Point(11.9, 47.4),
        Point(12.05, 47.55),
    ]
    return gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")


@pytest.fixture
def app_state():
    from app.state import AppState

    return AppState()


@pytest.fixture
def nuts_geojson_path(assets_dir):
    return assets_dir / "NUTS_RG_01M_2024_4326_LEVL_2.geojson"

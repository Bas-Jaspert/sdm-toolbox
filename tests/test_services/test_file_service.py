"""Tests for app.services.file_service."""

import io
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, mapping


# ---------------------------------------------------------------------------
# Helpers to build in-memory test files
# ---------------------------------------------------------------------------


def _make_geojson_bytes(points: list[tuple[float, float]], extra_cols: dict = None) -> bytes:
    """Build a minimal GeoJSON FeatureCollection."""
    features = []
    for lon, lat in points:
        props = {}
        if extra_cols:
            props.update(extra_cols)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    return json.dumps(fc).encode()


def _make_csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode()


def _make_zip_shapefile_bytes(points: list[tuple[float, float]], extra_cols: dict = None) -> bytes:
    """Create an in-memory ZIP containing a minimal shapefile."""
    gdf = gpd.GeoDataFrame(
        extra_cols or {},
        geometry=[Point(lon, lat) for lon, lat in points],
        crs="EPSG:4326",
    )
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        shp_path = Path(tmp) / "data.shp"
        gdf.to_file(shp_path)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                candidate = shp_path.with_suffix(ext)
                if candidate.exists():
                    zf.write(candidate, candidate.name)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_path_geojson(tmp_path):
    data = _make_geojson_bytes(
        [(11.5, 47.2), (11.6, 47.3)],
        extra_cols={"species": "Lagopus muta", "year": 2020},
    )
    p = tmp_path / "presences.geojson"
    p.write_bytes(data)
    return p


@pytest.fixture
def tmp_path_csv_lon_lat(tmp_path):
    rows = [
        {"lon": 11.5, "lat": 47.2, "species": "Lagopus muta", "year": 2020},
        {"lon": 11.6, "lat": 47.3, "species": "Lagopus muta", "year": 2021},
    ]
    p = tmp_path / "presences.csv"
    p.write_bytes(_make_csv_bytes(rows))
    return p


@pytest.fixture
def tmp_path_csv_decimal(tmp_path):
    rows = [
        {"decimalLongitude": 11.5, "decimalLatitude": 47.2},
        {"decimalLongitude": 11.6, "decimalLatitude": 47.3},
    ]
    p = tmp_path / "presences.csv"
    p.write_bytes(_make_csv_bytes(rows))
    return p


@pytest.fixture
def tmp_path_csv_xy(tmp_path):
    rows = [{"x": 11.5, "y": 47.2}, {"x": 11.6, "y": 47.3}]
    p = tmp_path / "presences.csv"
    p.write_bytes(_make_csv_bytes(rows))
    return p


@pytest.fixture
def tmp_path_csv_unknown_cols(tmp_path):
    rows = [{"alpha": 11.5, "beta": 47.2}]
    p = tmp_path / "presences.csv"
    p.write_bytes(_make_csv_bytes(rows))
    return p


@pytest.fixture
def tmp_path_zip_shp(tmp_path):
    data = _make_zip_shapefile_bytes(
        [(11.5, 47.2), (11.6, 47.3)],
        extra_cols={"species": ["Lagopus muta", "Lagopus muta"], "year": [2020, 2021]},
    )
    p = tmp_path / "presences.zip"
    p.write_bytes(data)
    return p


@pytest.fixture
def tmp_path_unsupported(tmp_path):
    p = tmp_path / "presences.xlsx"
    p.write_bytes(b"fake xlsx content")
    return p


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------


def _assert_schema(gdf: gpd.GeoDataFrame) -> None:
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert set(gdf.columns) >= {"species", "year", "geometry"}
    assert gdf.crs.to_epsg() == 4326
    assert all(isinstance(g, Point) for g in gdf.geometry)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_geojson(tmp_path_geojson):
    from app.services.file_service import parse_presence_file

    gdf = parse_presence_file(tmp_path_geojson)
    _assert_schema(gdf)
    assert len(gdf) == 2
    assert gdf["species"].iloc[0] == "Lagopus muta"


def test_parse_csv_lon_lat(tmp_path_csv_lon_lat):
    from app.services.file_service import parse_presence_file

    gdf = parse_presence_file(tmp_path_csv_lon_lat)
    _assert_schema(gdf)
    assert len(gdf) == 2
    assert abs(gdf.geometry.iloc[0].x - 11.5) < 1e-6
    assert abs(gdf.geometry.iloc[0].y - 47.2) < 1e-6


def test_parse_csv_decimal(tmp_path_csv_decimal):
    from app.services.file_service import parse_presence_file

    gdf = parse_presence_file(tmp_path_csv_decimal)
    _assert_schema(gdf)
    assert len(gdf) == 2


def test_parse_csv_xy(tmp_path_csv_xy):
    from app.services.file_service import parse_presence_file

    gdf = parse_presence_file(tmp_path_csv_xy)
    _assert_schema(gdf)
    assert len(gdf) == 2


def test_parse_zip_shapefile(tmp_path_zip_shp):
    from app.services.file_service import parse_presence_file

    gdf = parse_presence_file(tmp_path_zip_shp)
    _assert_schema(gdf)
    assert len(gdf) == 2


def test_detect_columns_returns_none(tmp_path_csv_unknown_cols):
    from app.services.file_service import detect_coord_columns

    df = pd.read_csv(tmp_path_csv_unknown_cols)
    result = detect_coord_columns(df)
    assert result is None


def test_parse_csv_missing_coord_raises(tmp_path_csv_unknown_cols):
    from app.services.file_service import parse_presence_file

    with pytest.raises(ValueError, match="coordinate"):
        parse_presence_file(tmp_path_csv_unknown_cols)


def test_species_fallback(tmp_path_csv_lon_lat, tmp_path):
    """File with no species column → species filled with 'unknown'."""
    from app.services.file_service import parse_presence_file

    rows = [{"lon": 11.5, "lat": 47.2}, {"lon": 11.6, "lat": 47.3}]
    p = tmp_path / "no_species.csv"
    p.write_bytes(_make_csv_bytes(rows))

    gdf = parse_presence_file(p)
    _assert_schema(gdf)
    assert all(gdf["species"] == "unknown")


def test_year_fallback(tmp_path):
    """File with no year column → year field is None."""
    from app.services.file_service import parse_presence_file

    rows = [{"lon": 11.5, "lat": 47.2}, {"lon": 11.6, "lat": 47.3}]
    p = tmp_path / "no_year.csv"
    p.write_bytes(_make_csv_bytes(rows))

    gdf = parse_presence_file(p)
    _assert_schema(gdf)
    assert all(pd.isna(gdf["year"]))


def test_unsupported_extension_raises(tmp_path_unsupported):
    from app.services.file_service import parse_presence_file

    with pytest.raises(ValueError, match="Unsupported"):
        parse_presence_file(tmp_path_unsupported)

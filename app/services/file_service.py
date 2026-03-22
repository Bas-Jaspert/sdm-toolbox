"""
File service module.

Parses local presence files (GeoJSON, shapefile ZIP, CSV/TXT) into the
standard species GeoDataFrame schema used by all SDM pipelines.
"""

import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# Ordered pairs of (lon_candidate_names, lat_candidate_names) tried in sequence.
# All comparisons are case-insensitive.
_COORD_CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("x", "y"),
    ("lon", "lat"),
    ("longitude", "latitude"),
    ("decimallongitude", "decimallatitude"),
]


def detect_coord_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    """Detect longitude/latitude column names from a DataFrame.

    Parameters
    ----------
    df:
        DataFrame whose columns are searched.

    Returns
    -------
    tuple[str, str] or None
        ``(lon_col, lat_col)`` using the original column names, or ``None``
        if no known pattern is found.
    """
    lower_to_original: dict[str, str] = {c.lower(): c for c in df.columns}
    for lon_lower, lat_lower in _COORD_CANDIDATE_PAIRS:
        if lon_lower in lower_to_original and lat_lower in lower_to_original:
            return lower_to_original[lon_lower], lower_to_original[lat_lower]
    return None


def parse_presence_file(path: Path) -> gpd.GeoDataFrame:
    """Parse a local presence file into the standard species GeoDataFrame schema.

    Supported formats:

    - ``.zip`` — ZIP archive containing a shapefile (``.shp`` + companions)
    - ``.geojson`` / ``.json`` — GeoJSON FeatureCollection
    - ``.csv`` / ``.txt`` — delimited text with coordinate columns

    Parameters
    ----------
    path:
        Path to the file to parse.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``["species", "year", "geometry"]``, CRS EPSG:4326.

    Raises
    ------
    ValueError
        If the file extension is not supported, the ZIP contains no shapefile,
        or coordinate columns cannot be found in a CSV/TXT file.
    """
    suffix = path.suffix.lower()
    if suffix == ".zip":
        gdf = _parse_zip_shapefile(path)
    elif suffix in (".geojson", ".json"):
        gdf = _parse_geojson(path)
    elif suffix in (".csv", ".txt"):
        gdf = _parse_csv(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Accepted: .zip (shapefile), .geojson, .json, .csv, .txt"
        )
    return _normalise_schema(gdf)


# ---------------------------------------------------------------------------
# Private parsers
# ---------------------------------------------------------------------------


def _parse_zip_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Extract a shapefile bundle from a ZIP and read it with geopandas."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp_path)

        shp_files = list(tmp_path.rglob("*.shp"))
        if not shp_files:
            raise ValueError(
                "ZIP archive contains no .shp file. "
                "Please include .shp, .shx, and .dbf in the archive."
            )
        return gpd.read_file(shp_files[0])


def _parse_geojson(path: Path) -> gpd.GeoDataFrame:
    """Read a GeoJSON file with geopandas."""
    return gpd.read_file(path)


def _parse_csv(path: Path) -> gpd.GeoDataFrame:
    """Read a CSV/TXT file and build Point geometries from coordinate columns."""
    df = pd.read_csv(path, sep=None, engine="python")
    result = detect_coord_columns(df)
    if result is None:
        raise ValueError(
            "Could not detect coordinate columns. "
            "Expected one of: x/y, lon/lat, longitude/latitude, "
            "decimalLongitude/decimalLatitude."
        )
    lon_col, lat_col = result
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    return gdf.drop(columns=[lon_col, lat_col])


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------


def _normalise_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalise a GeoDataFrame to the standard schema.

    Ensures columns ``["species", "year", "geometry"]`` with CRS EPSG:4326.
    Extra columns are dropped. Missing columns are filled with sensible defaults.
    """
    # Reproject if needed
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    cols_lower: dict[str, str] = {c.lower(): c for c in gdf.columns}

    # species
    species_src = next(
        (cols_lower[k] for k in ("species", "name", "taxon") if k in cols_lower),
        None,
    )
    gdf["species"] = gdf[species_src].astype(str) if species_src else "unknown"

    # year
    year_src = next(
        (cols_lower[k] for k in ("year", "date") if k in cols_lower),
        None,
    )
    if year_src:
        gdf["year"] = pd.to_numeric(gdf[year_src], errors="coerce").where(
            pd.notna(gdf[year_src]), other=None
        )
    else:
        gdf["year"] = None

    return gdf[["species", "year", "geometry"]].copy()

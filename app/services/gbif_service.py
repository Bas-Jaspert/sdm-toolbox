"""
GBIF service module.

Provides multiple modes for fetching species occurrence data from GBIF:
- "explore": Fast REST API query (up to 300 records per page)
- "deepdive": Paginated REST API query (all records, no limit)
- "own": Download GBIF dataset via authenticated API and cache as parquet
"""

import io
import zipfile
from pathlib import Path
from typing import Optional, Callable

import duckdb
import geopandas as gpd
import pandas as pd
import polars as pl
import requests
from pygbif import occurrences


def ensure_dataset_cached(
    key: str,
    user: str,
    pwd: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Check GBIF download status and cache as parquet.

    Uses occ.download_meta() to check if a previously created download is ready.
    If the file already exists in cache, returns immediately without checking.
    """
    if progress_callback is None:

        def noop(msg):
            pass

        progress_callback = noop

    cache_dir = Path.home() / ".sdm-toolbox" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.parquet"

    if cache_path.exists():
        progress_callback("Using cached dataset")
        return cache_path

    try:
        progress_callback("Checking download status...")

        status = occurrences.download_meta(key)
        download_status = status.get("status", "UNKNOWN")

        if download_status != "SUCCEEDED":
            raise ValueError(
                f"Download not ready (status: {download_status}). "
                f"Please create a download at gbif.org first."
            )

        progress_callback("Download ready. Fetching...")

        download_url = status.get("downloadLink")
        if not download_url:
            raise ValueError("No download link available")

        response = requests.get(download_url, timeout=300)
        response.raise_for_status()

        progress_callback("Extracting and processing...")

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            tsv_name = next(
                (
                    n
                    for n in zf.namelist()
                    if (n.endswith(".tsv") or n.endswith(".csv"))
                ),
                None,
            )
            if tsv_name is None:
                raise ValueError("No TSV file found in archive")

            with zf.open(tsv_name) as tsv_file:
                df = pl.read_csv(tsv_file, separator="\t", low_memory=True)

        lat_cols = ["decimalLatitude", "lat", "latitude", "y"]
        lon_cols = ["decimalLongitude", "lon", "longitude", "x"]

        lat_col = next((c for c in lat_cols if c in df.columns), None)
        lon_col = next((c for c in lon_cols if c in df.columns), None)

        if not lat_col or not lon_col:
            raise ValueError(
                f"Required coordinate columns not found. "
                f"Expected one of {lat_cols} and {lon_cols}"
            )

        df = df.rename({lat_col: "decimalLatitude", lon_col: "decimalLongitude"})

        if "species" not in df.columns and "scientificName" in df.columns:
            df = df.with_columns(pl.col("scientificName").alias("species"))
        if "year" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("year"))

        df.write_parquet(cache_path)
        progress_callback("Dataset cached successfully")
        return cache_path

    except Exception as e:
        raise ValueError(f"Failed to cache dataset {key}: {str(e)}") from e


def fetch_presences(
    mode: str,
    species: str,
    country: str,
    year_start: int,
    year_end: int,
    dataset_key: str = "",
    gbif_user: str = "",
    gbif_pwd: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> gpd.GeoDataFrame:
    """Fetch species occurrence data from GBIF in different modes."""
    if mode not in ["explore", "deepdive", "own"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'explore', 'deepdive', or 'own'."
        )

    if mode == "explore":
        return _fetch_explore(species, country, year_start, year_end)
    elif mode == "deepdive":
        return _fetch_deepdive(species, country, year_start, year_end)
    else:  # mode == "own"
        if not dataset_key:
            raise ValueError("dataset_key is required for 'own' mode")
        if not gbif_user or not gbif_pwd:
            raise ValueError("gbif_user and gbif_pwd are required for 'own' mode")
        return _fetch_own(
            dataset_key,
            species,
            year_start,
            year_end,
            gbif_user,
            gbif_pwd,
            progress_callback,
        )


def _fetch_explore(
    species: str, country: str, year_start: int, year_end: int
) -> gpd.GeoDataFrame:
    """Fast GBIF REST API query - up to 300 records."""
    params = {
        "scientificName": species,
        "country": country,
        "hasCoordinate": "true",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "limit": 300,
        "year": f"{year_start},{year_end}",
    }

    try:
        response = requests.get(
            "https://api.gbif.org/v1/occurrence/search", params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        records = data.get("results", [])
        if not records:
            return _empty_gdf()

        df = pd.json_normalize(records)

        if "species" not in df.columns:
            if "scientificName" in df.columns:
                df["species"] = df["scientificName"].copy()
            else:
                KeyError

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
            crs="EPSG:4326",
        )[["species", "year", "geometry"]]
        return _cleanup_gdf(gdf)

    except Exception as e:
        print(f"Error fetching data in explore mode: {e}")
        return _empty_gdf()


def _fetch_deepdive(
    species: str, country: str, year_start: int, year_end: int
) -> gpd.GeoDataFrame:
    """Paginated GBIF REST API query - all records."""
    all_records = []
    offset = 0
    limit = 300

    try:
        while True:
            result = occurrences.search(
                scientificName=species,
                country=country,
                hasCoordinate=True,
                basisOfRecord="HUMAN_OBSERVATION",
                year=f"{year_start},{year_end}",
                limit=limit,
                offset=offset,
            )

            records = result.get("results", [])
            if not records:
                break

            all_records.extend(records)

            if len(records) < limit:
                break

            offset += limit

        if not all_records:
            return _empty_gdf()

        df = pd.json_normalize(all_records)

        if "species" not in df.columns:
            if "scientificName" in df.columns:
                df["species"] = df["scientificName"].copy()
            else:
                df["species"] = species

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
            crs="EPSG:4326",
        )[["species", "year", "geometry"]]
        return _cleanup_gdf(gdf)

    except Exception as e:
        print(f"Error fetching data in deepdive mode: {e}")
        return _empty_gdf()


def _fetch_own(
    dataset_key: str,
    species: str,
    year_start: int,
    year_end: int,
    user: str,
    pwd: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> gpd.GeoDataFrame:
    """Load species data from cached parquet using duckdb for filtering."""
    try:
        parquet_path = ensure_dataset_cached(dataset_key, user, pwd, progress_callback)

        species_col = (
            "species"
            if "species" in pl.read_parquet(parquet_path, n_rows=1).columns
            else "scientificName"
        )

        query = f"""
            SELECT species, year, decimalLatitude, decimalLongitude
            FROM '{parquet_path}'
            WHERE LOWER({species_col}) LIKE '%' || LOWER('{species}') || '%'
            AND year >= {year_start} AND year <= {year_end}
        """

        df = duckdb.query(query).df()

        if df.empty:
            return _empty_gdf()

        gdf = gpd.GeoDataFrame(
            df[["species", "year"]],
            geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
            crs="EPSG:4326",
        )
        return _cleanup_gdf(gdf)

    except Exception as e:
        print(f"Error fetching data in own mode: {e}")
        return _empty_gdf()


def _empty_gdf() -> gpd.GeoDataFrame:
    """Return empty GeoDataFrame with correct schema."""
    return gpd.GeoDataFrame(
        columns=["species", "year", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _cleanup_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert non-geometry columns to strings for GeoJSON compatibility."""
    if gdf is None or gdf.empty:
        return gdf
    for col in gdf.columns:
        if col == "geometry":
            continue
        gdf[col] = gdf[col].apply(lambda x: str(x) if pd.notna(x) else None)
    return gdf

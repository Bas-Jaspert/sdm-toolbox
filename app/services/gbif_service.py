"""
GBIF service module.

Provides multiple modes for fetching species occurrence data from GBIF:
- "explore": Fast REST API query (up to 300 records per page)
- "deepdive": Paginated REST API query (all records, no limit)
- "own": Load from cached parquet dataset
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
import requests
from pygbif import occurrences


def ensure_dataset_cached(key: str) -> Path:
    """
    Download a GBIF dataset by key and cache it as parquet.

    Uses the pygbif library's occurrence download API to fetch a dataset.
    If the file already exists in cache, returns the path immediately
    without downloading.

    Parameters
    ----------
    key : str
        The GBIF dataset key (UUID).

    Returns
    -------
    Path
        Path to the cached parquet file at ~/.sdm-toolbox/datasets/<key>.parquet

    Raises
    ------
    ValueError
        If the dataset cannot be downloaded or processed.
    """
    cache_dir = Path.home() / ".sdm-toolbox" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.parquet"

    # Return immediately if already cached
    if cache_path.exists():
        return cache_path

    try:
        # Fetch all occurrences for this dataset
        # Use pagination to get all records
        all_records = []
        offset = 0
        limit = 300  # Max per API call

        while True:
            result = occurrences.search(
                datasetKey=key,
                limit=limit,
                offset=offset,
                hasCoordinate=True
            )

            records = result.get("results", [])
            if not records:
                break

            all_records.extend(records)

            # Check if there are more records
            if len(records) < limit:
                break

            offset += limit

        if not all_records:
            raise ValueError(f"No records found for dataset key: {key}")

        # Convert to DataFrame and filter required columns
        df = pd.json_normalize(all_records)

        # Ensure required columns exist
        required_cols = ["species", "year", "decimalLatitude", "decimalLongitude"]
        for col in required_cols:
            if col not in df.columns:
                if col == "species":
                    df[col] = df.get("scientificName", "")
                elif col == "year":
                    df[col] = pd.to_numeric(df.get("year", None), errors="coerce")
                elif col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset")

        # Save to parquet
        df.to_parquet(cache_path, index=False)
        return cache_path

    except Exception as e:
        raise ValueError(f"Failed to download/cache dataset {key}: {str(e)}") from e


def fetch_presences(
    mode: str,
    species: str,
    country: str,
    year: int,
    dataset_key: str = ""
) -> gpd.GeoDataFrame:
    """
    Fetch species occurrence data from GBIF in different modes.

    Parameters
    ----------
    mode : str
        One of "explore", "deepdive", or "own".
        - "explore": Fast GBIF REST API (~seconds, up to 300 records per page)
        - "deepdive": Paginated REST API query (all records, slower)
        - "own": Load from cached parquet dataset
    species : str
        Scientific name of the species.
    country : str
        ISO 3166-1 alpha-2 country code (e.g., "AT" for Austria).
    year : int
        Year of observations to filter by.
    dataset_key : str, optional
        Required for "own" mode. The GBIF dataset key (UUID).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns: species, year, geometry (Point), crs EPSG:4326.
        Returns empty GeoDataFrame if no records found (never returns None).

    Raises
    ------
    ValueError
        If mode is invalid or dataset_key is missing for "own" mode.
    """
    if mode not in ["explore", "deepdive", "own"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'explore', 'deepdive', or 'own'.")

    if mode == "explore":
        return _fetch_explore(species, country, year)
    elif mode == "deepdive":
        return _fetch_deepdive(species, country, year)
    else:  # mode == "own"
        if not dataset_key:
            raise ValueError("dataset_key is required for 'own' mode")
        return _fetch_own(dataset_key, species, year)


def _fetch_explore(species: str, country: str, year: int) -> gpd.GeoDataFrame:
    """
    Fast GBIF REST API query.

    Fetches up to 300 records in a single API call.
    """
    params = {
        "scientificName": species,
        "country": country,
        "hasCoordinate": "true",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "limit": 300,
    }

    try:
        response = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        occurrences_list = data.get("results", [])

        if not occurrences_list:
            # Return empty GeoDataFrame with correct schema
            return gpd.GeoDataFrame(
                columns=["species", "year", "geometry"],
                geometry="geometry",
                crs="EPSG:4326"
            )

        # Convert to DataFrame and extract required columns
        df = pd.json_normalize(occurrences_list)

        # Ensure species column exists
        if "species" not in df.columns:
            if "scientificName" in df.columns:
                df["species"] = df["scientificName"]
            else:
                df["species"] = species

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df[["species", "year"]],
            geometry=gpd.points_from_xy(
                df["decimalLongitude"],
                df["decimalLatitude"]
            ),
            crs="EPSG:4326"
        )

        return gdf

    except Exception as e:
        # Log error but return empty GeoDataFrame instead of raising
        print(f"Error fetching data in explore mode: {e}")
        return gpd.GeoDataFrame(
            columns=["species", "year", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )


def _fetch_deepdive(species: str, country: str, year: int) -> gpd.GeoDataFrame:
    """
    Paginated GBIF REST API query.

    Fetches all records by looping through pages until no more records found.
    """
    all_records = []
    offset = 0
    limit = 300  # Max per API call

    try:
        while True:
            result = occurrences.search(
                scientificName=species,
                country=country,
                hasCoordinate=True,
                basisOfRecord="HUMAN_OBSERVATION",
                year=year,
                limit=limit,
                offset=offset
            )

            records = result.get("results", [])
            if not records:
                break

            all_records.extend(records)

            # Check if there are more records
            if len(records) < limit:
                break

            offset += limit

        if not all_records:
            # Return empty GeoDataFrame with correct schema
            return gpd.GeoDataFrame(
                columns=["species", "year", "geometry"],
                geometry="geometry",
                crs="EPSG:4326"
            )

        # Convert to DataFrame
        df = pd.json_normalize(all_records)

        # Ensure species column exists
        if "species" not in df.columns:
            if "scientificName" in df.columns:
                df["species"] = df["scientificName"]
            else:
                df["species"] = species

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df[["species", "year"]],
            geometry=gpd.points_from_xy(
                df["decimalLongitude"],
                df["decimalLatitude"]
            ),
            crs="EPSG:4326"
        )

        return gdf

    except Exception as e:
        # Log error but return empty GeoDataFrame instead of raising
        print(f"Error fetching data in deepdive mode: {e}")
        return gpd.GeoDataFrame(
            columns=["species", "year", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )


def _fetch_own(dataset_key: str, species: str, year: int) -> gpd.GeoDataFrame:
    """
    Load species data from cached parquet dataset.

    Filters by species name (case-insensitive contains) and year.
    """
    try:
        # Get the cached parquet file
        parquet_path = ensure_dataset_cached(dataset_key)

        # Load the parquet file
        df = pd.read_parquet(parquet_path)

        # Filter by species (case-insensitive contains)
        if "species" in df.columns:
            species_mask = df["species"].str.lower().str.contains(
                species.lower(),
                na=False
            )
        elif "scientificName" in df.columns:
            df["species"] = df["scientificName"]
            species_mask = df["species"].str.lower().str.contains(
                species.lower(),
                na=False
            )
        else:
            # No species column, return empty
            return gpd.GeoDataFrame(
                columns=["species", "year", "geometry"],
                geometry="geometry",
                crs="EPSG:4326"
            )

        # Filter by year
        if "year" in df.columns:
            year_mask = df["year"] == year
        else:
            year_mask = True

        filtered_df = df[species_mask & year_mask]

        if filtered_df.empty:
            # Return empty GeoDataFrame with correct schema
            return gpd.GeoDataFrame(
                columns=["species", "year", "geometry"],
                geometry="geometry",
                crs="EPSG:4326"
            )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            filtered_df[["species", "year"]],
            geometry=gpd.points_from_xy(
                filtered_df["decimalLongitude"],
                filtered_df["decimalLatitude"]
            ),
            crs="EPSG:4326"
        )

        return gdf

    except Exception as e:
        # Log error but return empty GeoDataFrame instead of raising
        print(f"Error fetching data in own mode: {e}")
        return gpd.GeoDataFrame(
            columns=["species", "year", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )

"""Static metadata catalogue for all environmental predictor layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerMeta:
    """Metadata for a single environmental predictor layer.

    Parameters
    ----------
    description:
        Human-readable explanation of what the layer measures.
    units:
        Unit of measurement (e.g. ``"m"``, ``"°C × 10"``).
    data_source:
        Originating dataset or product name.
    category:
        Display category used for grouping in the UI.
    """

    description: str
    units: str
    data_source: str
    category: str


LAYER_CATALOGUE: dict[str, LayerMeta] = {
    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    "elevation": LayerMeta(
        "Elevation above sea level", "m", "USGS SRTMGL1", "Terrain"
    ),
    "slope": LayerMeta("Terrain slope angle", "°", "USGS SRTMGL1", "Terrain"),
    "aspect": LayerMeta("Slope aspect (azimuth)", "°", "USGS SRTMGL1", "Terrain"),
    "northness": LayerMeta(
        "Cosine of aspect — proxy for north-facing exposure",
        "−1–1",
        "USGS SRTMGL1",
        "Terrain",
    ),
    "eastness": LayerMeta(
        "Sine of aspect — proxy for east-facing exposure",
        "−1–1",
        "USGS SRTMGL1",
        "Terrain",
    ),
    # ------------------------------------------------------------------
    # Vegetation
    # ------------------------------------------------------------------
    "NDVI": LayerMeta(
        "Normalised Difference Vegetation Index",
        "−1–1",
        "Landsat 8 8-day composite",
        "Vegetation",
    ),
    "NARI": LayerMeta(
        "Narrowband Anthocyanin Reflectance Index — scrub/heath indicator",
        "index",
        "Sentinel-2",
        "Vegetation",
    ),
    "NCRI": LayerMeta(
        "Narrowband Carotenoid Reflectance Index — scrub/heath indicator",
        "index",
        "Sentinel-2",
        "Vegetation",
    ),
    "Trees": LayerMeta(
        "Canopy height masked to tree pixels", "m", "Meta Canopy Height", "Vegetation"
    ),
    "CHM": LayerMeta(
        "Canopy Height Model — full canopy height including shrubs",
        "m",
        "Meta Canopy Height",
        "Vegetation",
    ),
    # ------------------------------------------------------------------
    # Climate
    # ------------------------------------------------------------------
    "SWE": LayerMeta(
        "Snow Water Equivalent — water content of snowpack",
        "m",
        "ERA5-Land Hourly",
        "Climate",
    ),
    "snow_depth": LayerMeta(
        "Depth of snow on ground", "m", "ERA5-Land Hourly", "Climate"
    ),
    "snow_cover": LayerMeta(
        "Fraction of grid cell covered by snow", "0–1", "ERA5-Land Hourly", "Climate"
    ),
    "snow_albedo": LayerMeta(
        "Albedo of snow surface", "0–1", "ERA5-Land Hourly", "Climate"
    ),
    # ------------------------------------------------------------------
    # Land Cover & Human Impact
    # ------------------------------------------------------------------
    "landcover": LayerMeta(
        "CORINE land cover classification",
        "class code",
        "Copernicus Corine V20 100m 2018",
        "Land Cover & Human Impact",
    ),
    "GHMI": LayerMeta(
        "Global Human Modification Index — cumulative human pressure",
        "0–1",
        "GHM 2022 300m",
        "Land Cover & Human Impact",
    ),
    # ------------------------------------------------------------------
    # BioClim (WorldClim b1–b19)
    # ------------------------------------------------------------------
    "b1": LayerMeta(
        "Annual Mean Temperature", "°C × 10", "WorldClim BioClim", "BioClim"
    ),
    "b2": LayerMeta(
        "Mean Diurnal Range (mean of monthly max−min temperature)",
        "°C × 10",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b3": LayerMeta(
        "Isothermality (b2 / b7 × 100)", "%", "WorldClim BioClim", "BioClim"
    ),
    "b4": LayerMeta(
        "Temperature Seasonality (standard deviation × 100)",
        "°C × 100",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b5": LayerMeta(
        "Max Temperature of Warmest Month", "°C × 10", "WorldClim BioClim", "BioClim"
    ),
    "b6": LayerMeta(
        "Min Temperature of Coldest Month", "°C × 10", "WorldClim BioClim", "BioClim"
    ),
    "b7": LayerMeta(
        "Temperature Annual Range (b5 − b6)", "°C × 10", "WorldClim BioClim", "BioClim"
    ),
    "b8": LayerMeta(
        "Mean Temperature of Wettest Quarter",
        "°C × 10",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b9": LayerMeta(
        "Mean Temperature of Driest Quarter",
        "°C × 10",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b10": LayerMeta(
        "Mean Temperature of Warmest Quarter",
        "°C × 10",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b11": LayerMeta(
        "Mean Temperature of Coldest Quarter",
        "°C × 10",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b12": LayerMeta(
        "Annual Precipitation", "mm", "WorldClim BioClim", "BioClim"
    ),
    "b13": LayerMeta(
        "Precipitation of Wettest Month", "mm", "WorldClim BioClim", "BioClim"
    ),
    "b14": LayerMeta(
        "Precipitation of Driest Month", "mm", "WorldClim BioClim", "BioClim"
    ),
    "b15": LayerMeta(
        "Precipitation Seasonality (coefficient of variation)",
        "CV",
        "WorldClim BioClim",
        "BioClim",
    ),
    "b16": LayerMeta(
        "Precipitation of Wettest Quarter", "mm", "WorldClim BioClim", "BioClim"
    ),
    "b17": LayerMeta(
        "Precipitation of Driest Quarter", "mm", "WorldClim BioClim", "BioClim"
    ),
    "b18": LayerMeta(
        "Precipitation of Warmest Quarter", "mm", "WorldClim BioClim", "BioClim"
    ),
    "b19": LayerMeta(
        "Precipitation of Coldest Quarter", "mm", "WorldClim BioClim", "BioClim"
    ),
}

_CATEGORY_ORDER: list[str] = [
    "Terrain",
    "Vegetation",
    "Climate",
    "Land Cover & Human Impact",
    "BioClim",
]


def get_catalogue_by_category() -> dict[str, list[tuple[str, LayerMeta]]]:
    """Return the layer catalogue grouped by category in display order.

    Returns
    -------
    dict[str, list[tuple[str, LayerMeta]]]
        Ordered mapping of category name → list of (layer_name, meta) pairs.
    """
    grouped: dict[str, list[tuple[str, LayerMeta]]] = {
        cat: [] for cat in _CATEGORY_ORDER
    }
    for name, meta in LAYER_CATALOGUE.items():
        grouped[meta.category].append((name, meta))
    return grouped

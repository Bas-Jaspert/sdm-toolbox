"""Tests for soil layer metadata additions."""

from app.services.layer_metadata import (
    CATEGORY_NOTES,
    LAYER_CATALOGUE,
    TEMPORAL_LAYERS,
    _CATEGORY_ORDER,
    get_catalogue_by_category,
)

_SOIL_KEYS = ["soil_ph", "soil_soc", "soil_clay", "soil_sand", "soil_bdod", "soil_nitrogen"]


def test_soil_layers_in_catalogue() -> None:
    for key in _SOIL_KEYS:
        assert key in LAYER_CATALOGUE, f"{key!r} missing from LAYER_CATALOGUE"


def test_soil_category_order() -> None:
    assert "Soil Properties" in _CATEGORY_ORDER


def test_category_notes_soil() -> None:
    assert "Soil Properties" in CATEGORY_NOTES
    assert "ESDAC" in CATEGORY_NOTES["Soil Properties"]


def test_get_catalogue_by_category_includes_soil() -> None:
    grouped = get_catalogue_by_category()
    assert "Soil Properties" in grouped
    keys_in_group = [name for name, _ in grouped["Soil Properties"]]
    for key in _SOIL_KEYS:
        assert key in keys_in_group, f"{key!r} missing from grouped catalogue"


# ---------------------------------------------------------------------------
# Temporal flag tests
# ---------------------------------------------------------------------------

_EXPECTED_TEMPORAL = {"SWE", "snow_depth", "snow_cover", "snow_albedo", "NDVI", "NARI", "NCRI"}
_EXPECTED_STATIC = {
    "elevation", "slope", "aspect", "northness", "eastness",
    "Trees", "CHM", "landcover", "GHMI",
    "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
    "b10", "b11", "b12", "b13", "b14", "b15", "b16", "b17", "b18", "b19",
    "soil_ph", "soil_soc", "soil_clay", "soil_sand", "soil_bdod", "soil_nitrogen",
}


def test_temporal_layers_are_marked() -> None:
    for name in _EXPECTED_TEMPORAL:
        assert LAYER_CATALOGUE[name].temporal is True, f"{name} should be temporal"


def test_static_layers_are_not_temporal() -> None:
    for name in _EXPECTED_STATIC:
        assert LAYER_CATALOGUE[name].temporal is False, f"{name} should not be temporal"


def test_temporal_layers_frozenset() -> None:
    assert TEMPORAL_LAYERS == _EXPECTED_TEMPORAL

from app.services.layer_metadata import LAYER_CATALOGUE, get_catalogue_by_category

EXPECTED_CATEGORIES = {"Terrain", "Vegetation", "Climate", "Land Cover & Human Impact", "BioClim"}
KNOWN_LAYERS = [
    "elevation",
    "slope",
    "aspect",
    "northness",
    "eastness",
    "NDVI",
    "NARI",
    "NCRI",
    "Trees",
    "CHM",
    "SWE",
    "snow_depth",
    "snow_cover",
    "snow_albedo",
    "landcover",
    "GHMI",
    *[f"b{i}" for i in range(1, 20)],
]


class TestLayerCatalogue:
    def test_all_known_layers_present(self) -> None:
        for name in KNOWN_LAYERS:
            assert name in LAYER_CATALOGUE, f"Missing layer: {name}"

    def test_no_empty_fields(self) -> None:
        for name, meta in LAYER_CATALOGUE.items():
            assert meta.description, f"{name}: empty description"
            assert meta.units, f"{name}: empty units"
            assert meta.data_source, f"{name}: empty data_source"

    def test_get_catalogue_by_category_returns_expected_categories(self) -> None:
        grouped = get_catalogue_by_category()
        assert set(grouped.keys()) == EXPECTED_CATEGORIES

    def test_each_category_non_empty(self) -> None:
        for cat, items in get_catalogue_by_category().items():
            assert len(items) > 0, f"Category '{cat}' is empty"

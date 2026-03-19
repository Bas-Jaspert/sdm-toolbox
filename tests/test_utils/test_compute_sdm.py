import pytest
from toolbox.utils import compute_sdm


class TestComputeSDM:
    def test_compute_sdm_rf_returns_model(
        self, sample_presence_gdf, sample_background_gdf
    ):
        features = ["NDVI", "elevation", "slope"]
        model, results_df, ml_gdf = compute_sdm(
            presence=sample_presence_gdf,
            background=sample_background_gdf,
            features=features,
            model_type="Random Forest",
            n_trees=10,
            tree_depth=3,
            train_size=0.7,
        )

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_compute_sdm_returns_results_dataframe(
        self, sample_presence_gdf, sample_background_gdf
    ):
        features = ["NDVI", "elevation", "slope"]
        model, results_df, ml_gdf = compute_sdm(
            presence=sample_presence_gdf,
            background=sample_background_gdf,
            features=features,
            model_type="Random Forest",
            n_trees=10,
            tree_depth=3,
            train_size=0.7,
        )

        assert results_df is not None
        assert "roc_auc" in results_df.columns

    def test_compute_sdm_returns_ml_gdf(
        self, sample_presence_gdf, sample_background_gdf
    ):
        features = ["NDVI", "elevation", "slope"]
        model, results_df, ml_gdf = compute_sdm(
            presence=sample_presence_gdf,
            background=sample_background_gdf,
            features=features,
            model_type="Random Forest",
            n_trees=10,
            tree_depth=3,
            train_size=0.7,
        )

        assert ml_gdf is not None
        assert "PresAbs" in ml_gdf.columns
        assert len(ml_gdf) > 0

    def test_compute_sdm_maxent_returns_string_model(
        self, sample_presence_gdf, sample_background_gdf
    ):
        features = ["NDVI", "elevation", "slope"]
        model, results_df, ml_gdf = compute_sdm(
            presence=sample_presence_gdf,
            background=sample_background_gdf,
            features=features,
            model_type="Maxent",
            n_trees=10,
            tree_depth=3,
            train_size=0.7,
        )

        assert model == "Maxent"
        assert results_df.empty

    def test_compute_sdm_balanced_classes(
        self, sample_presence_gdf, sample_background_gdf
    ):
        features = ["NDVI", "elevation", "slope"]
        model, results_df, ml_gdf = compute_sdm(
            presence=sample_presence_gdf,
            background=sample_background_gdf,
            features=features,
            model_type="Random Forest",
            n_trees=10,
            tree_depth=3,
            train_size=0.7,
        )

        assert "PresAbs" in ml_gdf.columns
        assert (ml_gdf["PresAbs"] == 0).sum() > 0
        assert (ml_gdf["PresAbs"] == 1).sum() > 0

    def test_compute_sdm_raises_on_empty_presence(self, sample_background_gdf):
        import geopandas as gpd
        from shapely.geometry import Point

        empty_presence = gpd.GeoDataFrame(
            {"NDVI": [0.0], "elevation": [0.0], "slope": [0.0]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )

        features = ["NDVI", "elevation", "slope"]
        with pytest.raises(Exception):
            compute_sdm(
                presence=empty_presence,
                background=sample_background_gdf,
                features=features,
                model_type="Random Forest",
                n_trees=10,
                tree_depth=3,
                train_size=0.7,
            )

    def test_compute_sdm_column_sanitization(
        self, sample_presence_gdf, sample_background_gdf
    ):
        features = ["NDVI", "elevation", "slope"]
        model, results_df, ml_gdf = compute_sdm(
            presence=sample_presence_gdf,
            background=sample_background_gdf,
            features=features,
            model_type="Random Forest",
            n_trees=10,
            tree_depth=3,
            train_size=0.7,
        )

        for col in ml_gdf.columns:
            assert " " not in col

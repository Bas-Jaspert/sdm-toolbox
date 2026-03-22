import pytest
from app.state import AppState


class TestAppState:
    def test_app_state_defaults(self):
        state = AppState()

        assert state.species == ""
        assert state.country_code == ""
        assert state.county_name == ""
        assert state.year_start == 2015
        assert state.year_end == 2024
        assert state.data_mode == "explore"
        assert state.model_type == "rf"
        assert state.n_trees == 100
        assert state.max_depth == 3
        assert state.train_size == 0.75
        assert state.resolution == 30
        assert state.selected_layers == []
        assert state.layer_stack is None
        assert state.species_gdf is None
        assert state.model is None
        assert state.results_df is None
        assert state.classified_img is None

    def test_app_state_dataclass_fields(self):
        state = AppState()

        expected_fields = [
            "species",
            "country_code",
            "county_name",
            "year_start",
            "year_end",
            "data_mode",
            "dataset_key",
            "gbif_user",
            "gbif_pwd",
            "species_gdf",
            "selected_layers",
            "model_type",
            "n_trees",
            "max_depth",
            "train_size",
            "resolution",
            "layer_stack",
            "model",
            "results_df",
            "classified_img",
            "ml_gdf",
            "whatif_offsets",
        ]

        for field in expected_fields:
            assert hasattr(state, field)

    def test_state_data_mode_options(self):
        state = AppState()
        assert state.data_mode in ["explore", "deepdive", "own"]

    def test_state_model_type_options(self):
        state = AppState()
        assert state.model_type in ["rf", "maxent", "embedding"]

    def test_state_year_range(self):
        state = AppState()
        assert state.year_start < state.year_end
        assert state.year_start >= 2000
        assert state.year_end <= 2030

    def test_state_train_size_range(self):
        state = AppState()
        assert 0 < state.train_size < 1

    def test_state_n_trees_positive(self):
        state = AppState()
        assert state.n_trees > 0

    def test_state_max_depth_positive(self):
        state = AppState()
        assert state.max_depth > 0

    def test_state_resolution_default(self):
        state = AppState()
        assert state.resolution == 30
        assert state.resolution > 0

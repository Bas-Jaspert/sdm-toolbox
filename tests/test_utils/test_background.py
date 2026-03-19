import pytest
from pathlib import Path
from toolbox.utils import load_background_data


class TestLoadBackgroundData:
    def test_load_background_data_default_path(self):
        result = load_background_data()
        assert result is not None
        assert len(result) > 0

    def test_load_background_data_custom_path(self, background_csv_path):
        result = load_background_data(path=background_csv_path)
        assert result is not None
        assert len(result) > 0

    def test_load_background_data_returns_geodataframe(self, background_csv_path):
        result = load_background_data(path=background_csv_path)
        assert hasattr(result, "geometry")
        assert result.crs is not None

    def test_background_has_required_columns(self, background_csv_path):
        result = load_background_data(path=background_csv_path)
        required_columns = ["NDVI", "elevation", "slope"]
        for col in required_columns:
            assert col in result.columns

    def test_background_has_geometry_column(self, background_csv_path):
        result = load_background_data(path=background_csv_path)
        assert "geometry" in result.columns

    def test_background_no_null_geometries(self, background_csv_path):
        result = load_background_data(path=background_csv_path)
        assert result.geometry.is_valid.all()

    def test_load_background_data_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_background_data(path="/nonexistent/path.csv")

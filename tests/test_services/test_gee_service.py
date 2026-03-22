"""Tests for app.services.gee_service — load_custom_layer band naming."""

from unittest.mock import MagicMock, patch


def _make_ee_image_mock():
    """Return a mock ee.Image chain that records rename() calls."""
    img = MagicMock()
    selected = MagicMock()
    renamed = MagicMock()

    img.select.return_value = selected
    selected.rename.return_value = renamed

    return img, selected, renamed


def test_load_custom_layer_with_band_renames_to_display_name():
    """When a band name is supplied the returned image is renamed to *name*."""
    img, selected, renamed = _make_ee_image_mock()

    with patch("app.services.gee_service.ee") as mock_ee:
        mock_ee.Image.return_value = img
        from app.services.gee_service import load_custom_layer

        result = load_custom_layer("projects/foo/assets/bar", "slope_deg", "My Slope")

    img.select.assert_called_once_with("slope_deg")
    selected.rename.assert_called_once_with("My Slope")
    assert result is renamed


def test_load_custom_layer_without_band_renames_to_display_name():
    """When no band name is given (first band), the image is still renamed to *name*."""
    img, selected, renamed = _make_ee_image_mock()

    with patch("app.services.gee_service.ee") as mock_ee:
        mock_ee.Image.return_value = img
        from app.services.gee_service import load_custom_layer

        result = load_custom_layer("projects/foo/assets/bar", None, "My DEM")

    img.select.assert_called_once_with(0)
    selected.rename.assert_called_once_with("My DEM")
    assert result is renamed


def test_load_custom_layer_empty_band_string_renames_to_display_name():
    """Empty string band is treated the same as None — first band selected, renamed."""
    img, selected, renamed = _make_ee_image_mock()

    with patch("app.services.gee_service.ee") as mock_ee:
        mock_ee.Image.return_value = img
        import app.services.gee_service as gee_mod

        result = gee_mod.load_custom_layer("projects/foo/assets/bar", "", "My DEM")

    img.select.assert_called_once_with(0)
    selected.rename.assert_called_once_with("My DEM")
    assert result is renamed

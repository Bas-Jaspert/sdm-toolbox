"""Tests for resolution consistency in sdm_service pipelines."""

from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ee_chain():
    """Return a mock ee.Image-like chain that records method calls."""
    img = MagicMock()
    # Make each chained call return a new mock so assertions are independent.
    img.reproject.return_value = img
    img.clip.return_value = img
    img.classify.return_value = img
    return img


def _make_state(model_type: str = "rf", data_mode: str = "explore"):
    state = MagicMock()
    state.model_type = model_type
    state.data_mode = data_mode
    state.country_code = "AT"
    state.county_name = None
    state.n_trees = 100
    state.max_depth = 10
    state.train_size = 0.7
    state.selected_layers = ["elevation", "slope"]
    # layer_stack: each value is an independent mock Image
    state.layer_stack = {k: MagicMock() for k in state.selected_layers}
    # Each layer's reproject returns a sentinel mock
    for k, img in state.layer_stack.items():
        img.reproject.return_value = MagicMock(name=f"{k}_reprojected")
    state.species_gdf = MagicMock()
    state.species_gdf.shape = [10, 3]
    return state


# ---------------------------------------------------------------------------
# Issue 1: run_gee() must reproject predictors to 30 m before classify
# ---------------------------------------------------------------------------


def test_run_gee_reprojects_predictors_before_classify():
    """Predictor stack must be reprojected to EPSG:4326 / 30 m before .classify()."""
    state = _make_state(model_type="rf", data_mode="explore")

    predictor_img = _make_ee_chain()
    reprojected = _make_ee_chain()
    predictor_img.reproject.return_value = reprojected
    clipped = _make_ee_chain()
    reprojected.clip.return_value = clipped

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap") as mock_geemap,
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        # sampleRegions returns a FeatureCollection-like mock
        predictor_img.sampleRegions.return_value = MagicMock()
        # accuracy chain
        train_fc_mock = MagicMock()
        train_fc_mock.classify.return_value = MagicMock()
        train_fc_mock.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = 0.9
        predictor_img.sampleRegions.return_value.map.return_value = MagicMock()
        fc_mock = MagicMock()
        fc_mock.merge.return_value = train_fc_mock
        predictor_img.sampleRegions.return_value.map.return_value = fc_mock
        # presence / background merge
        bg_mock = MagicMock()
        bg_mock.map.return_value = MagicMock()
        predictor_img.sampleRegions.side_effect = [fc_mock, bg_mock]
        fc_mock.map.return_value = fc_mock
        bg_mock.map.return_value = bg_mock
        train_fc = MagicMock()
        fc_mock.merge.return_value = train_fc
        train_fc.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = 0.9
        # classifier chain
        classifier = MagicMock()
        classifier.setOutputMode.return_value = classifier
        classifier.explain.return_value.getInfo.return_value = {"importance": {}}
        rf_cls = MagicMock()
        rf_cls.setOutputMode.return_value = classifier
        rf_cls.train.return_value = rf_cls
        train_fc.classify.return_value = MagicMock()
        mock_ee.Classifier.smileRandomForest.return_value = MagicMock(
            **{
                "setOutputMode.return_value": MagicMock(
                    **{
                        "train.return_value": MagicMock(
                            **{
                                "setOutputMode.return_value": classifier,
                                "explain.return_value": MagicMock(
                                    getInfo=MagicMock(return_value={"importance": {}})
                                ),
                            }
                        )
                    }
                )
            }
        )
        mock_pd.DataFrame.return_value = MagicMock()

        from app.services.sdm_service import run_gee

        run_gee(state)

    # The predictor image must have been reprojected to EPSG:4326 at scale=30
    predictor_img.reproject.assert_called_once_with(crs="EPSG:4326", scale=30)
    # The reprojected image must then be clipped (not the original)
    reprojected.clip.assert_called()


# ---------------------------------------------------------------------------
# Issue 2: run_local() must pass a 30 m-reprojected layer dict to
#          get_species_features() so sampleRegions() uses a consistent scale
# ---------------------------------------------------------------------------


def test_run_local_passes_reprojected_layer_stack():
    """Each layer passed to get_species_features() must have been reprojected to 30 m."""
    state = _make_state(model_type="rf", data_mode="deepdive")

    reprojected_imgs = {k: MagicMock() for k in state.selected_layers}
    for k in state.selected_layers:
        state.layer_stack[k].reproject.return_value = reprojected_imgs[k]

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.get_species_features") as mock_gsf,
        patch("app.services.sdm_service.compute_sdm") as mock_compute,
        patch("app.services.sdm_service.classify_image_aoi") as mock_classify,
        patch("app.services.sdm_service.load_background_data") as mock_bg,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        presence_gdf = MagicMock()
        predictors = MagicMock()
        mock_gsf.return_value = (presence_gdf, predictors)
        mock_bg.return_value = MagicMock()
        model = MagicMock()
        results_df = MagicMock()
        ml_gdf = MagicMock()
        mock_compute.return_value = (model, results_df, ml_gdf)
        mock_classify.return_value = MagicMock()

        from app.services.sdm_service import run_local

        run_local(state)

    # Verify each layer was reprojected to EPSG:4326 at scale=30 before sampling
    for k in state.selected_layers:
        state.layer_stack[k].reproject.assert_called_once_with(
            crs="EPSG:4326", scale=30
        )

    # Verify get_species_features received the reprojected images, not the originals
    _, kwargs = mock_gsf.call_args
    passed_layer = kwargs.get("_layer") or mock_gsf.call_args[0][2]
    for k in state.selected_layers:
        assert passed_layer[k] is reprojected_imgs[k], (
            f"Layer '{k}' was not the reprojected image"
        )

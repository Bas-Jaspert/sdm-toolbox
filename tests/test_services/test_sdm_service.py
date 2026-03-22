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
    state.resolution = 30
    state.year_start = 2020
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
    passed_layer = kwargs["_layer"]
    for k in state.selected_layers:
        assert passed_layer[k] is reprojected_imgs[k], (
            f"Layer '{k}' was not the reprojected image"
        )


# ---------------------------------------------------------------------------
# Resolution: run_gee() must use state.resolution for all scale arguments
# ---------------------------------------------------------------------------


def test_run_gee_uses_state_resolution():
    """Both sampleRegions calls and the reproject call in run_gee() must use state.resolution."""
    state = _make_state(model_type="rf", data_mode="explore")
    state.resolution = 100

    predictor_img = _make_ee_chain()
    reprojected = _make_ee_chain()
    predictor_img.reproject.return_value = reprojected

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap"),
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        fc_mock, bg_mock = MagicMock(), MagicMock()
        predictor_img.sampleRegions.side_effect = [fc_mock, bg_mock]
        fc_mock.map.return_value = fc_mock
        bg_mock.map.return_value = bg_mock
        train_fc = MagicMock()
        fc_mock.merge.return_value = train_fc
        train_fc.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = 0.9
        rf = MagicMock()
        rf.setOutputMode.return_value.train.return_value.setOutputMode.return_value = MagicMock(
            explain=MagicMock(
                return_value=MagicMock(getInfo=MagicMock(return_value={"importance": {}}))
            )
        )
        mock_ee.Classifier.smileRandomForest.return_value = rf
        mock_pd.DataFrame.return_value = MagicMock()

        from app.services.sdm_service import run_gee

        run_gee(state)

    # Both sampleRegions calls must use scale=100
    for i, c in enumerate(predictor_img.sampleRegions.call_args_list):
        scale_used = c.kwargs.get("scale")
        assert scale_used == 100, f"sampleRegions call {i}: expected scale=100, got {scale_used}"

    # reproject must use scale=100
    predictor_img.reproject.assert_called_once_with(crs="EPSG:4326", scale=100)


# ---------------------------------------------------------------------------
# Resolution: run_local() pre-reproject must use state.resolution
# ---------------------------------------------------------------------------


def test_run_local_uses_state_resolution():
    """Pre-reproject loop in run_local() must use state.resolution."""
    state = _make_state(model_type="rf", data_mode="deepdive")
    state.resolution = 100

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
        mock_gsf.return_value = (MagicMock(), MagicMock())
        mock_bg.return_value = MagicMock()
        mock_compute.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_classify.return_value = MagicMock()

        from app.services.sdm_service import run_local

        run_local(state)

    for k in state.selected_layers:
        state.layer_stack[k].reproject.assert_called_once_with(
            crs="EPSG:4326", scale=100
        )


# ---------------------------------------------------------------------------
# Resolution: run_embedding() must ignore state.resolution (always scale=30)
# ---------------------------------------------------------------------------


def test_run_embedding_ignores_resolution():
    """All three sampleRegions calls in run_embedding() must use scale=30 regardless of state.resolution."""
    import pandas as _pd  # real pandas for eval_df; avoid shadowing the patched pd

    state = _make_state(model_type="embedding", data_mode="explore")
    state.resolution = 100

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap") as mock_geemap,
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.roc_auc_score") as mock_roc,
        patch("app.services.sdm_service.pd") as mock_pd,
    ):
        mock_aoi.return_value = (MagicMock(), None)

        # Set up the mosaic chain (mosaic is what sampleRegions is called on at line 303)
        mosaic = MagicMock()
        mock_ee.ImageCollection.return_value.filter.return_value.mosaic.return_value = mosaic

        # dot_product_img = mosaic.multiply(...).reduce(...).divide(...).rename(...)
        # This is the mock that sampleRegions is called on at lines 343 and 347.
        dot_product_img = (
            mosaic.multiply.return_value.reduce.return_value.divide.return_value.rename.return_value
        )

        # Return a valid DataFrame so eval_df.empty / column checks pass
        mock_geemap.ee_to_df.return_value = _pd.DataFrame(
            {"score": [0.8, 0.6], "PresAbs": [1, 0]}
        )
        mock_roc.return_value = 0.85
        mock_pd.DataFrame.return_value = MagicMock()

        from app.services.sdm_service import run_embedding

        run_embedding(state)

    # sampleRegions on mosaic (line 303) must use scale=30
    assert mosaic.sampleRegions.call_count == 1
    assert mosaic.sampleRegions.call_args.kwargs["scale"] == 30, (
        f"mosaic.sampleRegions: expected scale=30, got {mosaic.sampleRegions.call_args.kwargs.get('scale')}"
    )

    # sampleRegions on dot_product_img (lines 343, 347) must both use scale=30
    assert dot_product_img.sampleRegions.call_count == 2
    for i, c in enumerate(dot_product_img.sampleRegions.call_args_list):
        scale_used = c.kwargs.get("scale")
        assert scale_used == 30, (
            f"dot_product_img.sampleRegions call {i}: expected scale=30, got {scale_used}"
        )


# ---------------------------------------------------------------------------
# Maxent: run_gee() must train in PROBABILITY mode (the only GEE-supported mode),
# threshold probability at 0.5 for the accuracy step, and select("probability")
# from the classified AOI image
# ---------------------------------------------------------------------------


def _setup_maxent_run_gee_mocks(
    mock_ee: MagicMock, predictor_img: MagicMock
) -> tuple[MagicMock, MagicMock]:
    """Wire mocks for a Maxent run_gee() call.

    Returns
    -------
    train_fc : MagicMock
        The merged training FeatureCollection mock.
    maxent_base : MagicMock
        The object returned by ``ee.Classifier.amnhMaxent()``.
    """
    fc_mock, bg_mock = MagicMock(), MagicMock()
    predictor_img.sampleRegions.side_effect = [fc_mock, bg_mock]
    fc_mock.map.return_value = fc_mock
    bg_mock.map.return_value = bg_mock
    train_fc = MagicMock()
    fc_mock.merge.return_value = train_fc
    # Direct classify chain (used by RF; Maxent goes through .map() first)
    train_fc.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = (
        0.85
    )
    # Post-threshold chain: classify(...).map(...).errorMatrix(...)
    train_fc.classify.return_value.map.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = (
        0.85
    )
    maxent_base = MagicMock()
    mock_ee.Classifier.amnhMaxent.return_value = maxent_base
    return train_fc, maxent_base


def test_run_gee_maxent_does_not_set_classification_mode():
    """amnhMaxent() must NOT be called with setOutputMode('CLASSIFICATION') — GEE rejects it."""
    state = _make_state(model_type="maxent", data_mode="explore")
    predictor_img = _make_ee_chain()

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap"),
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        mock_pd.DataFrame.return_value = MagicMock()
        _train_fc, maxent_base = _setup_maxent_run_gee_mocks(mock_ee, predictor_img)

        from app.services.sdm_service import run_gee

        run_gee(state)

    # setOutputMode must never be called on amnhMaxent() itself; only PROBABILITY
    # is supported and the classifier defaults to it without an explicit call.
    maxent_base.setOutputMode.assert_not_called()


def test_run_gee_maxent_thresholds_probability_for_accuracy():
    """Classified training FC must have .map() called to add a 'classification' property."""
    state = _make_state(model_type="maxent", data_mode="explore")
    predictor_img = _make_ee_chain()

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap"),
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        mock_pd.DataFrame.return_value = MagicMock()
        train_fc, _base = _setup_maxent_run_gee_mocks(mock_ee, predictor_img)

        from app.services.sdm_service import run_gee

        run_gee(state)

    # Maxent outputs "probability" (float); .map() must be used to derive
    # the discrete "classification" property for errorMatrix.
    train_fc.classify.return_value.map.assert_called_once()


def test_run_gee_maxent_selects_probability_band():
    """The classified AOI image must have .select('probability') called on it."""
    state = _make_state(model_type="maxent", data_mode="explore")
    predictor_img = _make_ee_chain()

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap"),
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        mock_pd.DataFrame.return_value = MagicMock()
        _setup_maxent_run_gee_mocks(mock_ee, predictor_img)

        from app.services.sdm_service import run_gee

        run_gee(state)

    # _make_ee_chain() chains reproject/clip/classify back to predictor_img,
    # so .select("probability") is asserted on the same object.
    predictor_img.select.assert_called_with("probability")

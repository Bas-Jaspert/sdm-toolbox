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
# Resolution: run_gee() must use state.resolution for sampleRegions calls
# ---------------------------------------------------------------------------


def test_run_gee_uses_state_resolution():
    """Both sampleRegions calls in run_gee() must use state.resolution."""
    state = _make_state(model_type="rf", data_mode="explore")
    state.resolution = 100

    predictor_img = _make_ee_chain()

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
        all_fc = MagicMock()
        fc_mock.merge.return_value = all_fc
        all_fc.randomColumn.return_value.filter.return_value.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = 0.9
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
        The train-split FeatureCollection mock (after randomColumn + filter).
    maxent_base : MagicMock
        The object returned by ``ee.Classifier.amnhMaxent()``.
    """
    fc_mock, bg_mock = MagicMock(), MagicMock()
    predictor_img.sampleRegions.side_effect = [fc_mock, bg_mock]
    fc_mock.map.return_value = fc_mock
    bg_mock.map.return_value = bg_mock
    # all_fc = presence.merge(background).randomColumn(seed=42)
    all_fc = MagicMock()
    fc_mock.merge.return_value = all_fc
    # train_fc = all_fc.filter(lt(0.75)); both filter calls return the same mock
    train_fc = all_fc.randomColumn.return_value.filter.return_value
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
    # the discrete "classification" property for errorMatrix (and also for
    # the AUC eval rename), so it is called at least once.
    train_fc.classify.return_value.map.assert_called()


def _make_state_with_years(years: list[int], temporal_layers: list[str] | None = None):
    """State mock with a real GeoDataFrame carrying per-point years."""
    import geopandas as _gpd
    from shapely.geometry import Point

    state = _make_state(model_type="rf", data_mode="explore")
    if temporal_layers is not None:
        state.selected_layers = temporal_layers
        state.layer_stack = {k: MagicMock() for k in temporal_layers}
    state.species_gdf = _gpd.GeoDataFrame(
        {"species": ["Sp"] * len(years), "year": years},
        geometry=[Point(float(i), float(i)) for i in range(len(years))],
        crs="EPSG:4326",
    )
    return state


def test_run_gee_calls_get_layer_info_per_unique_year():
    """get_layer_information must be called once per unique observation year
    when temporal layers are selected."""
    state = _make_state_with_years([2020, 2020, 2021, 2022], temporal_layers=["NDVI"])

    predictor_img = _make_ee_chain()

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap"),
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
        patch("app.services.sdm_service.gee_service") as mock_gee,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        year_samples = MagicMock()
        year_samples.filter.return_value = year_samples
        predictor_img.sampleRegions.return_value = year_samples
        year_samples.map.return_value = year_samples
        flattened = MagicMock()
        mock_ee.FeatureCollection.return_value.flatten.return_value = flattened
        flattened.size.return_value.getInfo.return_value = 4
        flattened.map.return_value = flattened
        bg_mock = MagicMock()
        bg_mock.map.return_value = bg_mock
        predictor_img.sampleRegions.side_effect = None
        predictor_img.sampleRegions.return_value = year_samples
        all_fc = MagicMock()
        flattened.merge.return_value = all_fc
        all_fc.randomColumn.return_value.filter.return_value.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = 0.9
        rf = MagicMock()
        rf.setOutputMode.return_value.train.return_value.setOutputMode.return_value = MagicMock(
            explain=MagicMock(
                return_value=MagicMock(
                    getInfo=MagicMock(return_value={"importance": {}})
                )
            )
        )
        mock_ee.Classifier.smileRandomForest.return_value = rf
        mock_pd.DataFrame.return_value = MagicMock()
        mock_gee.get_layer_information.return_value = {"NDVI": MagicMock()}

        from app.services.sdm_service import run_gee
        run_gee(state)

    assert mock_gee.get_layer_information.call_count == 3
    mock_gee.get_layer_information.assert_any_call(2020)
    mock_gee.get_layer_information.assert_any_call(2021)
    mock_gee.get_layer_information.assert_any_call(2022)


def test_run_gee_no_temporal_layers_skips_loop():
    """When only static layers are selected, get_layer_information must NOT
    be called (fast path)."""
    state = _make_state_with_years([2020, 2021], temporal_layers=["elevation"])

    predictor_img = _make_ee_chain()

    with (
        patch("app.services.sdm_service.get_aoi_from_nuts") as mock_aoi,
        patch("app.services.sdm_service.geemap"),
        patch("app.services.sdm_service.ee") as mock_ee,
        patch("app.services.sdm_service.pd") as mock_pd,
        patch("app.services.sdm_service.gee_service") as mock_gee,
    ):
        mock_aoi.return_value = (MagicMock(), None)
        mock_ee.Image.cat.return_value = predictor_img
        fc_mock, bg_mock = MagicMock(), MagicMock()
        predictor_img.sampleRegions.side_effect = [fc_mock, bg_mock]
        fc_mock.map.return_value = fc_mock
        bg_mock.map.return_value = bg_mock
        all_fc = MagicMock()
        fc_mock.merge.return_value = all_fc
        all_fc.randomColumn.return_value.filter.return_value.classify.return_value.errorMatrix.return_value.accuracy.return_value.getInfo.return_value = 0.9
        rf = MagicMock()
        rf.setOutputMode.return_value.train.return_value.setOutputMode.return_value = MagicMock(
            explain=MagicMock(
                return_value=MagicMock(
                    getInfo=MagicMock(return_value={"importance": {}})
                )
            )
        )
        mock_ee.Classifier.smileRandomForest.return_value = rf
        mock_pd.DataFrame.return_value = MagicMock()

        from app.services.sdm_service import run_gee
        run_gee(state)

    mock_gee.get_layer_information.assert_not_called()


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

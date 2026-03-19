"""
SDM service module.

Provides three SDM pipeline functions:
- run_gee(): GEE-native pipeline (Explore / Own Dataset modes, server-side)
- run_local(): sklearn pipeline (Deep Dive mode, data downloaded to Python)
- run_embedding(): Dot-product embedding pipeline (Google Satellite Embedding V1, server-side)
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
from app.state import AppState
from toolbox.utils import (
    get_aoi_from_nuts,
    get_species_features,
    compute_sdm,
    classify_image_aoi,
    load_background_data,
)
import geemap
import ee
import pandas as pd
from sklearn.metrics import roc_auc_score

_BG_PATH: Path = Path(__file__).parents[2] / "assets" / "background_data.csv"


def _cleanup_gdf_for_gee(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    """Convert non-geometry columns to strings for GeoJSON/EE compatibility."""
    if gdf is None or gdf.empty:
        return gdf
    for col in gdf.columns:
        if col == "geometry":
            continue
        gdf[col] = gdf[col].apply(lambda x: str(x) if pd.notna(x) else None)
    return gdf


def run_gee(state: AppState) -> AppState:
    """
    GEE-native SDM pipeline for Explore and Own Dataset modes.

    All computation stays server-side; no data is downloaded to Python.
    Supports Random Forest and Maxent classifiers.

    Parameters
    ----------
    state : AppState
        Current application state.  Must have species_gdf, country_code,
        selected_layers, layer_stack, model_type, n_trees, and max_depth set.

    Returns
    -------
    AppState
        Updated state with model, classified_img, and results_df populated.

    Raises
    ------
    RuntimeError
        If the GEE pipeline fails for any reason.
    """
    try:
        # 1. Get AOI
        country_aoi, county_aoi = get_aoi_from_nuts(
            state.country_code, state.county_name or None
        )
        aoi = county_aoi if county_aoi is not None else country_aoi

        # 1b. Clean up species_gdf for GeoJSON compatibility
        state.species_gdf = _cleanup_gdf_for_gee(state.species_gdf)

        # 2. Convert species GeoDataFrame to EE FeatureCollection
        presence_fc = geemap.gdf_to_ee(state.species_gdf)

        # 3. Sample background points in GEE within the AOI (1:1 ratio)
        n_presence = state.species_gdf.shape[0]
        background_fc = ee.FeatureCollection.randomPoints(
            region=aoi,
            points=n_presence,
        )

        # 4. Stack predictor layers
        predictors = ee.Image.cat([state.layer_stack[k] for k in state.selected_layers])

        # 5. Sample features at presence and background points (server-side)
        presence_samples = predictors.sampleRegions(
            collection=presence_fc,
            properties=[],
            scale=30,
        )
        background_samples = predictors.sampleRegions(
            collection=background_fc,
            properties=[],
            scale=30,
        )

        # 6. Add PresAbs property (1 for presence, 0 for background)
        presence_samples = presence_samples.map(lambda f: f.set("PresAbs", 1))
        background_samples = background_samples.map(lambda f: f.set("PresAbs", 0))
        train_fc = presence_samples.merge(background_samples)

        # 7. Train in CLASSIFICATION mode so errorMatrix gets discrete labels
        if state.model_type == "rf":
            classifier_cls = (
                ee.Classifier.smileRandomForest(
                    numberOfTrees=state.n_trees,
                    maxNodes=state.max_depth,
                    bagFraction=0.8,
                )
                .setOutputMode("CLASSIFICATION")
                .train(train_fc, "PresAbs", state.selected_layers)
            )
        elif state.model_type == "maxent":
            classifier_cls = ee.Classifier.amnhMaxent().train(
                train_fc, "PresAbs", state.selected_layers
            )
        else:
            raise ValueError(
                f"run_gee() does not support model_type='{state.model_type}'. "
                "Use 'rf' or 'maxent'."
            )

        # 8. Accuracy from discrete classifications
        classified_train = train_fc.classify(classifier_cls)
        accuracy = (
            classified_train.errorMatrix("PresAbs", "classification")
            .accuracy()
            .getInfo()
        )

        # 9. Switch to PROBABILITY for the suitability map (RF only)
        classifier = (
            classifier_cls.setOutputMode("PROBABILITY")
            if state.model_type == "rf"
            else classifier_cls
        )

        # 10. Classify AOI
        classified_img = predictors.clip(aoi).classify(classifier)

        # 11. Extract feature importances from GEE RF (best-effort)
        results_data: dict = {"overall_accuracy": [accuracy]}
        if state.model_type == "rf":
            try:
                importance = classifier_cls.explain().getInfo().get("importance", {})
                for feat, imp in importance.items():
                    results_data[feat] = [imp]
            except Exception:
                pass  # importance not available; accuracy-only results are fine

        # 12. Store results
        state.model = classifier
        state.classified_img = classified_img
        state.results_df = pd.DataFrame(results_data)

    except Exception as e:
        raise RuntimeError(f"GEE SDM pipeline failed: {str(e)}") from e

    return state


def run_local(state: AppState) -> AppState:
    """
    sklearn SDM pipeline for Deep Dive mode.

    Downloads sampled feature data to Python and trains a local sklearn model,
    then re-applies it server-side via GEE for classification.

    Parameters
    ----------
    state : AppState
        Current application state.  Must have species_gdf, country_code,
        selected_layers, layer_stack, model_type, n_trees, max_depth,
        and train_size set.

    Returns
    -------
    AppState
        Updated state with model, classified_img, and results_df populated.

    Raises
    ------
    RuntimeError
        If the local SDM pipeline fails for any reason.
    """
    try:
        # 1. Get AOI
        country_aoi, county_aoi = get_aoi_from_nuts(
            state.country_code, state.county_name or None
        )
        aoi = county_aoi if county_aoi is not None else country_aoi

        # 2. Extract features (downloads data to Python)
        presence_gdf, predictors = get_species_features(
            _species_gdf=state.species_gdf,
            features=state.selected_layers,
            _layer=state.layer_stack,
        )

        # 3. Load background data
        background_gdf = load_background_data(path=_BG_PATH)

        # Map short model_type keys to the string names used by compute_sdm()
        model_type_map = {
            "rf": "Random Forest",
            "maxent": "Maxent",
            "embedding": "Embedding",
        }

        # 4. Train local model
        model, results_df, ml_gdf = compute_sdm(
            presence=presence_gdf,
            background=background_gdf,
            features=state.selected_layers,
            model_type=model_type_map[state.model_type],
            n_trees=state.n_trees,
            tree_depth=state.max_depth,
            train_size=state.train_size,
        )

        # 5. Classify AOI
        classified_img = classify_image_aoi(
            image=predictors,
            aoi=aoi,
            ml_gdf=ml_gdf,
            model=model,
            features=state.selected_layers,
        )

        # 6. Store results
        state.model = model
        state.results_df = results_df
        state.classified_img = classified_img
        state.ml_gdf = ml_gdf

    except Exception as e:
        raise RuntimeError(f"Local (sklearn) SDM pipeline failed: {str(e)}") from e

    return state


def run_embedding(state: AppState) -> AppState:
    """
    Dot-product embedding SDM pipeline using Google's Satellite Embedding V1.

    Computes cosine similarity between the mean presence embedding vector and
    every pixel in the AOI as a habitat suitability score.  Entirely server-side
    except for a small sample pulled to Python to compute ROC-AUC.

    Parameters
    ----------
    state : AppState
        Current application state.  Must have species_gdf, country_code,
        county_name, and year set.

    Returns
    -------
    AppState
        Updated state with model, classified_img, and results_df populated.

    Raises
    ------
    RuntimeError
        If the embedding SDM pipeline fails for any reason.
    """
    try:
        # 1. Load annual embedding mosaic for state.year_start
        # Use year_start if in valid range, otherwise fall back to latest (2023)
        embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

        selected_year = state.year_start
        if selected_year < 2015 or selected_year > 2023:
            selected_year = 2023  # Fallback to latest available

        mosaic = embeddings.filter(
            ee.Filter.date(f"{selected_year}-01-01", f"{selected_year + 1}-01-01")
        ).mosaic()

        # 2a. Clean up species_gdf for GeoJSON compatibility
        state.species_gdf = _cleanup_gdf_for_gee(state.species_gdf)

        # 2. Convert species GeoDataFrame to EE FeatureCollection
        presence_fc = geemap.gdf_to_ee(state.species_gdf)
        n_presence = state.species_gdf.shape[0]

        # 2b. Get AOI for sampling
        country_aoi, county_aoi = get_aoi_from_nuts(
            state.country_code, state.county_name or None
        )
        aoi = county_aoi if county_aoi is not None else country_aoi

        # 3. Sample embedding vectors at all presence points
        presence_samples = mosaic.sampleRegions(
            collection=presence_fc,
            properties=[],
            scale=30,
        )

        # 4. Compute per-band mean across all presence samples → mean embedding vector
        band_names = mosaic.bandNames()
        mean_dict = presence_samples.reduceColumns(
            reducer=ee.Reducer.mean().repeat(band_names.size()),
            selectors=band_names,
        )
        mean_values = ee.List(mean_dict.get("mean"))

        # 5. Build constant image from mean vector
        mean_image = ee.Image.constant(mean_values).rename(band_names)

        # 6. Dot product: mosaic · mean_image summed across bands → raw suitability raster
        dot_product_img = mosaic.multiply(mean_image).reduce(ee.Reducer.sum())

        # 7. Normalize to cosine similarity: divide by |mosaic| (L2 norm per pixel)
        mosaic_norm = mosaic.pow(2).reduce(ee.Reducer.sum()).sqrt()
        # Guard against divide-by-zero with a small epsilon
        dot_product_img = dot_product_img.divide(
            mosaic_norm.max(ee.Image.constant(1e-9))
        )

        # 8. Sample dot product scores at presence + background points in GEE
        n_sample = min(100, n_presence)  # Use up to 100 points for AUC eval

        presence_sample_fc = presence_fc.randomColumn().limit(n_sample, "random")

        background_sample_fc = ee.FeatureCollection.randomPoints(
            region=aoi,
            points=n_sample,
        )

        # 9. Pull to Python and compute ROC-AUC
        # Rename band to "score" for a predictable column name in the DataFrame.
        dot_product_img = dot_product_img.rename("score")
        presence_scores_fc = dot_product_img.sampleRegions(
            collection=presence_sample_fc, properties=[], scale=30
        ).map(lambda f: f.set("PresAbs", 1))
        background_scores_fc = dot_product_img.sampleRegions(
            collection=background_sample_fc, properties=[], scale=30
        ).map(lambda f: f.set("PresAbs", 0))
        eval_fc = presence_scores_fc.merge(background_scores_fc)

        eval_df = geemap.ee_to_df(eval_fc)
        if eval_df.empty or "score" not in eval_df.columns:
            raise RuntimeError(
                "Embedding evaluation returned no samples — cannot compute AUC."
            )
        auc = roc_auc_score(eval_df["PresAbs"], eval_df["score"])

        # 10. Clip the dot product image to AOI (already computed above)
        classified_img = dot_product_img.clip(aoi)

        # 11. Store results
        state.model = "embedding"
        state.results_df = pd.DataFrame({"roc_auc": [auc]})
        state.classified_img = classified_img

    except Exception as e:
        raise RuntimeError(f"Embedding SDM pipeline failed: {str(e)}") from e

    return state

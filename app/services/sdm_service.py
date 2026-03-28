"""
SDM service module.

Provides two SDM pipeline functions:
- run_gee(): GEE-native pipeline (all data modes, server-side)
- run_embedding(): Dot-product embedding pipeline (Google Satellite Embedding V1, server-side)
"""

from app.state import AppState
from app.services import gee_service
from app.services.gbif_service import _cleanup_gdf as _cleanup_gdf_for_gee
from app.services.layer_metadata import TEMPORAL_LAYERS
from toolbox.utils import get_aoi_from_nuts
import geemap
import ee
import pandas as pd
from sklearn.metrics import roc_auc_score


def run_gee(state: AppState) -> AppState:
    """
    GEE-native SDM pipeline for all data modes.

    All computation stays server-side; only a small eval sample (~200 scores)
    is downloaded to Python to compute ROC-AUC.  Supports Random Forest and
    Maxent classifiers.

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

        # 4. Split selected layers into temporal (year-varying) and static
        selected_temporal = [name for name in state.selected_layers if name in TEMPORAL_LAYERS]
        selected_static = [name for name in state.selected_layers if name not in TEMPORAL_LAYERS]

        # 5a. Stack predictor layers for AOI classification map and background
        #     sampling — always uses state.layer_stack (loaded at year_start).
        predictors = ee.Image.cat([state.layer_stack[k] for k in state.selected_layers])

        # 5b. Sample presence points — per-observation-year when temporal layers
        #     are selected, single-image fast path otherwise.
        excluded_count = 0
        if not selected_temporal:
            presence_samples = predictors.sampleRegions(
                collection=presence_fc,
                properties=[],
                scale=state.resolution,
            )
        else:
            static_img = (
                ee.Image.cat([state.layer_stack[name] for name in selected_static])
                if selected_static
                else None
            )
            unique_years = sorted(
                int(y) for y in state.species_gdf["year"].dropna().unique()
            )
            per_year: list = []
            for year in unique_years:
                year_layers = gee_service.get_layer_information(year)
                temporal_img = ee.Image.cat([year_layers[name] for name in selected_temporal])
                year_predictors = (
                    ee.Image.cat([static_img, temporal_img])
                    if static_img is not None
                    else temporal_img
                )
                year_fc = presence_fc.filter(ee.Filter.eq("year", year))
                samples = year_predictors.sampleRegions(
                    collection=year_fc,
                    properties=[],
                    scale=state.resolution,
                ).filter(ee.Filter.notNull(selected_temporal))
                per_year.append(samples)

            presence_samples = ee.FeatureCollection(per_year).flatten()
            excluded_count = n_presence - presence_samples.size().getInfo()

        # 5c. Sample background points from year_start layer stack (unchanged)
        background_samples = predictors.sampleRegions(
            collection=background_fc,
            properties=[],
            scale=state.resolution,
        )

        # 6. Add PresAbs labels and split 75 / 25 for train / holdout eval.
        #    Using randomColumn ensures a reproducible, stratified-like split without
        #    an extra GEE round-trip.
        presence_samples = presence_samples.map(lambda f: f.set("PresAbs", 1))
        background_samples = background_samples.map(lambda f: f.set("PresAbs", 0))
        all_fc = presence_samples.merge(background_samples).randomColumn(seed=42)
        train_fc = all_fc.filter(ee.Filter.lt("random", 0.75))
        eval_fc = all_fc.filter(ee.Filter.gte("random", 0.75))

        # 7. Train classifier on the 75 % split.
        #    RF uses CLASSIFICATION mode so errorMatrix gets discrete labels.
        #    Maxent only supports PROBABILITY mode (GEE constraint), so we
        #    threshold at 0.5 server-side before calling errorMatrix.
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

        # 8. Overall accuracy on the training split (discrete labels).
        classified_train = train_fc.classify(classifier_cls)
        if state.model_type == "maxent":
            classified_train = classified_train.map(
                lambda f: f.set(
                    "classification",
                    f.getNumber("probability").gte(0.5).toInt(),
                )
            )
        accuracy = (
            classified_train.errorMatrix("PresAbs", "classification")
            .accuracy()
            .getInfo()
        )

        # 9. Switch to PROBABILITY for the suitability map and AUC eval.
        classifier = classifier_cls.setOutputMode("PROBABILITY")

        # 10. Classify AOI — no forced reproject; GEE uses adaptive pyramid rendering.
        if state.model_type == "rf":
            classified_img = predictors.clip(aoi).classify(classifier)
        elif state.model_type == "maxent":
            classified_img = (
                predictors.clip(aoi).classify(classifier).select("probability")
            )

        # 11. ROC-AUC on the 25 % holdout eval set.
        #     Rename the probability output to "score" for a uniform column name,
        #     then download only that tiny FeatureCollection (~1 KB).
        results_data: dict = {
            "overall_accuracy": [accuracy],
            "excluded_presence_count": [excluded_count],
        }
        try:
            if state.model_type == "rf":
                eval_scored = eval_fc.classify(classifier).map(
                    lambda f: f.set("score", f.getNumber("classification"))
                )
            else:
                eval_scored = eval_fc.classify(classifier).map(
                    lambda f: f.set("score", f.getNumber("probability"))
                )
            eval_df = geemap.ee_to_df(eval_scored.select(["PresAbs", "score"]))
            if not eval_df.empty and "score" in eval_df.columns:
                results_data["roc_auc"] = [
                    roc_auc_score(eval_df["PresAbs"], eval_df["score"])
                ]
        except Exception:
            pass  # AUC not critical; overall_accuracy is always present

        # 12. Extract feature importances from GEE RF (best-effort).
        if state.model_type == "rf":
            try:
                importance = classifier_cls.explain().getInfo().get("importance", {})
                for feat, imp in importance.items():
                    results_data[feat] = [imp]
            except Exception:
                pass

        # 13. Store results
        state.model = classifier
        state.classified_img = classified_img
        state.results_df = pd.DataFrame(results_data)

    except Exception as e:
        raise RuntimeError(f"GEE SDM pipeline failed: {str(e)}") from e

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
            scale=30,  # fixed at native embedding tile resolution
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

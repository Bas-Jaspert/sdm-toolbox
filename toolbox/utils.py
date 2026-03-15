import math
import tempfile
import json
import random
import requests
import pandas as pd
import geopandas as gpd
import geemap
import shapely
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import ee
import matplotlib

from sklearn.inspection import permutation_importance
from sklearn.utils.fixes import parse_version


def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=100, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()
    translation_dict = {f"b{i + 1}": f"BIO{(i + 1):02d}" for i in range(19)} | {
        "elevation": "Geländehöhe",
        "slope": "Hangneigung",
        "aspect": "Exposition",
        "landcover": "CLC",
        "snow_cover": "Schneebedeckung",
        "snow_depth": "Schneetiefe",
    }
    # `labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
    # renamed to `tick_labels`. The following code handles this, but as a
    # scikit-learn user you probably can write simpler code by using `labels=...`
    # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).
    tick_labels_parameter_name = (
        "tick_labels"
        if parse_version(matplotlib.__version__) >= parse_version("3.9")
        else "labels"
    )
    tick_labels_dict = {
        tick_labels_parameter_name: [
            translation_dict.get(key) if key in translation_dict.keys() else key
            for key in X.columns[perm_sorted_idx].tolist()
        ]
    }
    ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band.

    Parameters
    ----------
        image (ee.Image): A Sentinel-2 image.

    Returns
    -------
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask).divide(10000)


def calculate_nari(img):
    """Calcultate Near Reflectance Index from Earth Engine Image.

    Parameters
    ----------
    img : ee.Image
        An Iamge containing Green and Red Edge 2 bands.

    Returns
    -------
    nari: ee.Image
        An Image containing the Nari index.
    """
    green = img.select("B3")
    red_edge_1 = img.select("B5")
    nari = (
        (ee.Image.constant(1).divide(green)).subtract(
            ee.Image.constant(1).divide(red_edge_1)
        )
    ).divide(
        (ee.Image.constant(1).divide(green)).add(
            ee.Image.constant(1).divide(red_edge_1)
        )
    )
    return nari


def calculate_ncri(img):
    """Calcultate Near Reflectance Index from Earth Engine Image.

    Parameters
    ----------
    img : ee.Image
        An Iamge containing Green and Red Edge 2 bands.

    Returns
    -------
    nari: ee.Image
        An Image containing the NCRI index.
    """
    red_edge_1 = img.select("B5")
    red_edge_3 = img.select("B7")
    ncri = (
        (ee.Image.constant(1).divide(red_edge_1)).subtract(
            ee.Image.constant(1).divide(red_edge_3)
        )
    ).divide(
        (ee.Image.constant(1).divide(red_edge_1)).add(
            ee.Image.constant(1).divide(red_edge_3)
        )
    )
    return ncri


def gee_calculate_scrub_index(index: str = None, year: int = None):
    """Calculate the NARI or NCRI.

    Parameters
    ----------
    index : str, optional
        One of 'nari' or 'ncri', by default None
    year : int, optional
        Year to calculate the index for, by default None

    Returns
    -------
    ee.Image
        Image containing the requested index.
    """

    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{year}-09-01", f"{year}-11-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 65))
        .map(mask_s2_clouds)
        .select(["B3", "B5", "B7"])
    )
    if index == "nari":
        nari = sentinel.map(calculate_nari)
        return nari.median()
    elif index == "ncri":
        ncri = sentinel.map(calculate_ncri)
        return ncri.median()
    else:
        print("ERROR. No layer returned")


def calculate_ndvi(year: int = None):
    """Calcultate Normalized Difference Vegetation Index
    from Earth Engine Image.

    Parameters
    ----------
    img : ee.Image
        An Iamge containing Red and NIR bands.

    Returns
    -------
    nari: ee.Image
        An Image containing the Nari index.
    """
    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 65))
        .map(mask_s2_clouds)
        .select(["B3", "B5", "B7"])
    )
    red = sentinel.select("B4")
    nir = sentinel.select("B8")
    ndvi = (red.subtract(nir)).divide(red.add(nir)).mean().rename("NDVI")
    return ndvi


def get_era5_snowfall(date, poi):
    """Get the aggregated snowfall of the last 4 days for a given date
    and Point of Interest.

    Parameters
    ----------
    date : str
        Date in 'YYYY-MM-DD' fomrat
    poi : ee.FeatureCollection
        A FeatureCollection containing the Points of Interest.

    Returns
    -------
    _type_
        _description_
    """
    window_end = ee.Date(date)
    window_start = window_end.advance(ee.Number(-4), "days")
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(window_start, window_end)
        .select("snowfall_sum")
        .sum()
        .multiply(1000)
    )
    return era5.sampleRegions(collection=poi, properties=["name", "datetime"])


def get_era5_snowcover(date, poi):
    """Get the aggregated snow cover of the last 4 days for a given date
    and Point of Interest.

    Parameters
    ----------
    date : str
        Date in 'YYYY-MM-DD' fomrat
    poi : ee.FeatureCollection
        A FeatureCollection containing the Points of Interest.

    Returns
    -------
    _type_
        _description_
    """
    window_end = ee.Date(date)
    window_start = window_end.advance(ee.Number(-4), "days")
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(window_start, window_end)
        .select("snow_cover")
        .mean()
    )
    return era5.sampleRegions(collection=poi, properties=["name", "datetime"])


def get_species_data(species_name, country_code, database: str = "gbif"):
    """
    Retrieves observational data for a specific species using the
    GBIF API and returns it as a pandas DataFrame.

    Parameters
    ----------
        species_name : str
            The scientific name of the species to query.
        country_code : str
            Country code where the observation data will be queried.

    Returns
    -------
        pd.DataFrame
            A pandas DataFrame containing the observational data.
    """
    if database == "iNaturalist":
        user_id = input("Type in User-ID: ")

    match database:
        case "gbif":
            base_url = "https://api.gbif.org/v1/occurrence/search"
        case "iNaturalist":
            base_url = (
                "https://api.inaturalist.org/v1/observations?"
                f"user_id={user_id}&quality_grade=needs_id&rank=genus"
            )
    params = {
        "scientificName": species_name,
        "country": country_code,
        "hasCoordinate": "true",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "limit": 10000,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        occurrences = data.get("results", [])

        if occurrences:  # If data is present
            df = pd.json_normalize(occurrences)
            return gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude),
                crs="EPSG:4326",
            )[["species", "year", "month", "geometry"]]
        else:
            print("No data found for the given speciesand country code.")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def remove_duplicates(data, grain_size, aoi):
    """Remove duplicate georefernced points from data within
    an Area of Interest and a given resolution.

    Parameters
    ----------
    data : _type_
        _description_
    grain_size : _type_
        _description_
    aoi : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Select one occurrence record per pixel at the chosen spatial resolution
    random_raster = ee.Image.random().reproject("EPSG:4326", None, grain_size).clip(aoi)
    rand_point_vals = random_raster.sampleRegions(
        collection=ee.FeatureCollection(data), geometries=True
    )
    return rand_point_vals.distinct("random")


def get_aoi_from_nuts(country_code: str = "AT", county_name: str = None):
    NUTS_0 = gpd.read_file(r"./assets/NUTS_RG_01M_2024_4326_LEVL_0.geojson")
    NUTS_2 = gpd.read_file(r"./assets/NUTS_RG_01M_2024_4326_LEVL_2.geojson")
    country = geemap.gdf_to_ee(NUTS_0.loc[NUTS_0.CNTR_CODE == country_code])
    if county_name:
        county = geemap.gdf_to_ee(NUTS_2.loc[NUTS_2.NUTS_NAME == county_name])
    else:
        county = None

    return country, county


def get_layer_information(year: int):
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filter(
        ee.Filter.date(f"{year}-01-01", f"{year + 1}-01-01")
    )
    BIOCLIM = {
        key: (ee.Image("projects/ee-sebasd1991/assets/BioClim").select(key))
        for key in ee.Image("projects/ee-sebasd1991/assets/BioClim")
        .bandNames()
        .getInfo()
    }
    terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003"))
    canopyHeight = ee.ImageCollection(
        "projects/sat-io/open-datasets/facebook/meta-canopy-height"
    ).mosaic()
    layer = {
        "landcover": ee.Image("COPERNICUS/CORINE/V20/100m/2018").select("landcover"),
        "elevation": ee.Image("USGS/SRTMGL1_003").select("elevation"),
        "slope": terrain.select("slope"),
        "aspect": terrain.select("aspect"),
        "northness": terrain.select("aspect")
        .multiply(math.pi / 180)
        .cos()
        .rename("northness"),
        "eastness": terrain.select("aspect")
        .multiply(math.pi / 180)
        .sin()
        .rename("eastness"),
        "NDVI": (
            ee.ImageCollection("LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mean()
        ),
        "GHMI": ee.ImageCollection("projects/sat-io/open-datasets/GHM/HM_2022_300M")
        .first()
        .rename("GHMI"),
        "Trees": canopyHeight.updateMask(canopyHeight.gte(1)).rename("Trees"),
        "SWE": era5.select(["snow_depth_water_equivalent"], ["swe"])
        .mean()
        .rename("SWE"),
        "snow_depth": era5.select(["snow_depth"], ["snow_depth"]).mean(),
        "snow_cover": era5.select(["snow_cover"], ["snow_cover"]).mean(),
        "snow_albedo": era5.select(["snow_albedo"], ["snow_albedo"]).mean(),
        "CHM": canopyHeight.rename("CHM"),
        "NARI": gee_calculate_scrub_index("nari", year).rename("NARI"),
        "NCRI": gee_calculate_scrub_index("ncri", year).rename("NCRI"),
    }
    return {**layer, **BIOCLIM}


def get_layer_visualization_params(layer_name: str):
    try:
        paletteHM = [
            "#4c6100",
            "#adda25",
            "#e2ff9b",
            "#ffff73",
            "#ffe629",
            "#ffd37f",
            "#ffaa00",
            "#e69808",
            "#e60000",
            "#a80000",
            "#730000",
        ]
        paletteTrees = ["#4A2354", "#fde725"]
        vis_params = {
            "elevation": {
                "min": 0,
                "max": 4000,
                "palette": geemap.colormaps.palettes["terrain"],
            },
            "slope": {
                "min": 0,
                "max": 60,
                "palette": geemap.colormaps.palettes["viridis"],
            },
            "NDVI": {
                "min": -1,
                "max": 1,
                "palette": geemap.colormaps.palettes["RdYlGn"],
            },
            "CHM": {
                "min": 0,
                "max": 25,
                "palette": geemap.colormaps.palettes["viridis"],
            },
            "Trees": {"min": 1, "max": 1, "palette": paletteTrees},
            "NARI": {
                "min": -1,
                "max": 1,
                "palette": geemap.colormaps.palettes["RdYlGn"],
            },
            "NCRI": {
                "min": -1,
                "max": 1,
                "palette": geemap.colormaps.palettes["RdYlGn"],
            },
            "GHMI": {"min": 0, "max": 1, "palette": paletteHM},
            "SWE": {
                "min": 0,
                "max": 500,
                "palette": geemap.colormaps.palettes["Blues"],
            },
            "snow_depth": {
                "min": 0,
                "max": 200,
                "palette": geemap.colormaps.palettes["Blues"],
            },
            "snow_cover": {
                "min": 0,
                "max": 100,
                "palette": geemap.colormaps.palettes["Blues"],
            },
            "snow_albedo": {
                "min": 0,
                "max": 1,
                "palette": geemap.colormaps.palettes["Greys"],
            },
            "b1": {
                "min": -30,
                "max": 50,
                "palette": geemap.colormaps.palettes["viridis"],
            },
            "aspect": {
                "min": 0,
                "max": 360,
                "palette": geemap.colormaps.palettes["hsv"],
            },
            "northness": {
                "min": -1,
                "max": 1,
                "palette": geemap.colormaps.palettes["coolwarm"],
            },
            "eastness": {
                "min": -1,
                "max": 1,
                "palette": geemap.colormaps.palettes["coolwarm"],
            },
        }
        return vis_params.get(layer_name, {})
    except KeyError:
        return {}


def plot_correlation_heatmap(dataframe, h_size=10, show_labels=False):
    # Calculate Spearman correlation coefficients
    correlation_matrix = dataframe.corr(method="spearman")

    # Create a heatmap
    plt.figure(figsize=(h_size, h_size - 2))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")

    # Optionally display values on the heatmap
    if show_labels:
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{correlation_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    columns = dataframe.columns.tolist()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Variables Correlation Matrix")
    plt.colorbar(label="Spearman Correlation")
    plt.savefig("correlation_heatmap_plot.png")
    plt.show()


def load_background_data(path=r"./assets/background_data.csv"):
    """Loads background data from file.
    The file is expected to be a CSV with a '.geo' column containing geometries in GeoJSON format.

    Args:
        path (pathlike|str, optional): _description_. Defaults to r"../assets/background_data.csv".

    Returns:
        gpd.GeoDataFrame: Returns a GeoDataFrame containing the background data.
    """
    bg_df = pd.read_csv(path, sep=",")
    s = bg_df[".geo"].astype(str).str.replace("'", '"').apply(json.loads)
    geoms = s.apply(shapely.geometry.shape)
    bg_gdf = gpd.GeoDataFrame(
        bg_df.drop([".geo"], axis=1), geometry=geoms, crs="epsg:4326"
    )

    return bg_gdf


def get_species_features(
    _species_gdf: gpd.GeoDataFrame = None, features: list = None, _layer: dict = None
):
    if _layer is None and features is None:
        raise ValueError(" 'layer' and 'features' must be provided.")
    else:
        predictors = ee.Image.cat([_layer[feature] for feature in features])
        presence_gdf = geemap.ee_to_gdf(
            predictors.sampleRegions(
                collection=geemap.gdf_to_ee(_species_gdf), geometries=True
            )
        )
        return presence_gdf, predictors


def compute_sdm(
    presence: gpd.GeoDataFrame = None,
    background: gpd.GeoDataFrame = None,
    features: list = None,
    model_type: str = "Random Forest",
    n_trees: int = 100,
    tree_depth: int = 5,
    train_size: float = 0.7,
):
    """_summary_

    Parameters
    ----------
    presence : gpd.GeoDataFrame, optional
        _description_, by default None
    background : gpd.GeoDataFrame, optional
        _description_, by default None
    features : list, optional
        _description_, by default None
    model_type : str, optional
        _description_, by default "Random Forest"
    n_trees : int, optional
        _description_, by default 100
    tree_depth : int, optional
        _description_, by default 5
    train_size : float, optional
        _description_, by default 0.7

    Returns
    -------
    _type_
        _description_
    """
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import (
        train_test_split,
        KFold,
        StratifiedKFold,
        StratifiedShuffleSplit,
        cross_val_score,
        GridSearchCV,
    )
    from sklearn.feature_selection import RFE, RFECV, SelectFromModel
    from sklearn.inspection import PartialDependenceDisplay, permutation_importance

    # if model_type== "Maxent" & (backgrund_gdf is None):
    #     background_gdf = load_background_data()[features+['geometry']]
    # else:
    #     background_gdf = load_background_data()[features+['geometry']].sample(n=species_gdf.shape[0], axis=0)
    translation_dict = {f"b{i + 1}": f"BIO{(i + 1):02d}" for i in range(19)} | {
        "elevation": "Geländehöhe",
        "slope": "Hangneigung",
        "aspect": "Exposition",
        "landcover": "CLC",
        "snow_cover": "Schneebedeckung",
        "snow_depth": "Schneetiefe",
    }

    background = (
        background[features + ["geometry"]].copy().sample(n=presence.shape[0], axis=0)
    )

    background["PresAbs"] = 0
    presence["PresAbs"] = 1
    presence = presence[background.columns]

    ml_gdf = pd.concat([background, presence], axis=0).reset_index(drop=True)
    ml_gdf.columns = [_.replace(" ", "_") for _ in ml_gdf.columns]

    y = ml_gdf["PresAbs"]
    X = ml_gdf[features]

    match model_type:
        case "Random Forest":
            results = []

            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=train_size, stratify=y, shuffle=True
                )
                model = RandomForestClassifier(
                    n_estimators=n_trees,
                    max_samples=0.8,
                    min_samples_leaf=0.1,
                    verbose=0,
                    class_weight="balanced",
                    max_depth=tree_depth,
                )
                model.fit(X_train, y_train)
                results.append(
                    [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])]
                    + model.feature_importances_.tolist()
                )

            results_df = pd.DataFrame(results, columns=["roc_auc"] + X.columns.tolist())

            mdi_importances = pd.Series(
                model.feature_importances_,
                index=[
                    translation_dict.get(key) if key in translation_dict.keys() else key
                    for key in X[features].columns.tolist()
                ],
            )
            tree_importance_sorted_idx = np.argsort(model.feature_importances_)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=500)
            mdi_importances.sort_values().plot.barh(ax=ax1)
            ax1.set_xlabel("Gini importance")
            plot_permutation_importance(model, X_test[features], y_test, ax2)
            ax2.set_xlabel("Verringerung der Korrektklassifikationsrate")
            fig.suptitle(
                "Impurity-based vs. Permutation Importances bei multikollinearen Features"
            )
            _ = fig.tight_layout()

        case "Maxent":
            model = "Maxent"
            results_df = pd.DataFrame()
        case "Embedding":
            model = "Embedding"
            embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            mosaic = embeddings.filter(
                ee.Filter.date("2024-01-01", "2025-01-01")
            ).mosaic()
            results_df = pd.DataFrame()
            # results_df = mosaic.sampleRegions(
            #     collection= geemap.gdf_to_ee(ml_gdf),
            #     properties = ['PresAbs'],
            #     scale= 30
            # )
    # partial dependence
    # from sklearn.inspection import PartialDependenceDisplay, permutation_importance

    # # fig, ax = plt.subplots(figsize=(12, 8))
    # _, ax = plt.subplots(ncols=4, nrows=2, figsize=(9, 8), constrained_layout=True)
    # ax = ax.flatten()[:-1]  # remove the last subplot

    # PartialDependenceDisplay.from_estimator(
    #     rf, X_train, features=X.columns.to_list(), kind='average', ax=ax
    # )
    return model, results_df, ml_gdf


def classify_image_aoi(image, aoi, ml_gdf, model, features):
    # if "classified_img_pr" in st.session_state:
    #     del st.session_state.classified_img_pr
    fc = geemap.gdf_to_ee(ml_gdf)
    seed = random.randint(1, 1000)
    tr_presence_points = (
        fc.filter(ee.Filter.eq("PresAbs", 1)).randomColumn(seed=seed).sort("random")
    )
    tr_pseudo_abs_points = (
        fc.filter(ee.Filter.eq("PresAbs", 0))
        .randomColumn(seed=seed)
        .sort("random")
        .limit(tr_presence_points.size().getInfo())
    )
    train_pvals = tr_presence_points.merge(tr_pseudo_abs_points)
    if isinstance(model, RandomForestClassifier):
        # Random Forest classifier
        classifier = ee.Classifier.smileRandomForest(
            seed=seed,
            numberOfTrees=model.n_estimators,
            maxNodes=model.max_depth,
            # shrinkage=0.1, # gradient
            # variablesPerSplit=None,
            minLeafPopulation=round(train_pvals.size().getInfo() * 0.1, 0),  # rf
            bagFraction=0.8,  # rf
        )
        classifier_pr = classifier.setOutputMode("PROBABILITY").train(
            train_pvals, "PresAbs", features
        )
        classified_img_pr = (
            image.reproject(crs="EPSG:4326", scale=30).clip(aoi).classify(classifier_pr)
        )
        return classified_img_pr

    if model == "Maxent":
        classifier = ee.Classifier.amnhMaxent()

        # Presence probability: Habitat suitability map
        classifier_pr = classifier.train(train_pvals, "PresAbs", features)
        classified_img_pr = (
            image.reproject(crs="EPSG:4326", scale=30).clip(aoi).classify(classifier_pr)
        )
        return classified_img_pr.select("probability")
    if model == "Embedding":
        embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        mosaic = embeddings.filter(ee.Filter.date("2024-01-01", "2025-01-01")).mosaic()
        sampleEmbeddings = mosaic.sampleRegions(
            collection=geemap.gdf_to_ee(ml_gdf), properties=["PresAbs"], scale=30
        )
        tr_presence_points = sampleEmbeddings.filter(ee.Filter.eq("PresAbs", 1))
        tr_pseudo_abs_points = sampleEmbeddings.filter(ee.Filter.eq("PresAbs", 0))
        train_pvals = tr_presence_points.merge(tr_pseudo_abs_points)

        # Random Forest classifier
        classifier = ee.Classifier.smileRandomForest(
            seed=seed,
            numberOfTrees=500,
            maxNodes=5,
            # shrinkage=0.1, # gradient
            # variablesPerSplit=None,
            minLeafPopulation=round(train_pvals.size().getInfo() * 0.1, 0),  # rf
            bagFraction=0.8,  # rf
        )
        # Presence probability: Habitat suitability map
        classifier_pr = classifier.setOutputMode("PROBABILITY").train(
            train_pvals, "PresAbs", mosaic.bandNames()
        )
        classified_img_pr = mosaic.clip(aoi).classify(classifier_pr)
        return classified_img_pr


def plot_hier_clustering(dataframe):
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    from collections import defaultdict

    X = dataframe.copy().drop("geometry", axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    _ = fig.tight_layout()

    cluster_ids = hierarchy.fcluster(dist_linkage, 0.35, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = X.columns[selected_features]
    selected_features_names

    return fig, selected_features_names


def get_index_info():
    index_info = {
        "landcover": "Land Cover (CLC) Datensätze werden im Rahmen des europäischen Erdbeobachtungsprogramms Copernicus von der Europäischen Umweltagentur (EEA) bereitgestellt. Das Projekt dient der flächendeckenden Erfassung, Klassifizierung und Harmonisierung von Landbedeckungs- und Landnutzungsdaten innerhalb Europas.Der Datensatz weist eine Mindestkartiereinheit von 25 ha und eine thematische Genauigkeit von mindestens 85 % auf. Die Klassifikation erfolgt nach einem hierarchisch aufgebauten System mit 44 Landbedeckungsklassen, die in fünf Hauptkategorien gegliedert sind: künstliche Flächen, landwirtschaftliche Flächen, Wälder und naturnahe Flächen, Feuchtgebiete sowie Gewässer.",
        "GHMI": "Global Human Modification Index quantifies the extent of human modification.",
        "NDVI": "Einer der bekanntesten Vegetationindizes, welcher sich aus dem Quotienten der Differenz und der Summe des Nahinfrarot und rot Bandes, also der reflektierten Strahlung zwischen 700 bis 1100 nm für Nahinfrarot und 400 bis 700 nm für Rot berechnet:   $NDVI=\\frac{NIR-R}{NIR+R}$",
        "NARI": "Normalized Anthoyanin Reflectance Index (NARI), welcher sensitiv auf den Anthocyanin Gehalt der Vegetationsdecke reagiert:   $NARI=\\frac{\\frac{1}{G}-\\frac{1}{Red Edge 1}}{\\frac{1}{G}+\\frac{1}{Red Edge 1}}$.",
        "NCRI": "Normalized Chlorophyll Refelectance Index (NCRI), welcher Informationen über den Chlorophyll Gehalt erfasst: $NCRI=\\frac{\\frac{1}{Red Edge 1}-\\frac{1}{Red Edge 3}}{\\frac{1}{Red Edge 1}+\\frac{1}{Red Edge 3}}$.",
        "CHM": "Canopy Height Model (CHM) stellt die Höhe der Vegetationsdecke über dem Boden dar und wird durch die Differenz zwischen dem Digitalen Oberflächenmodell (DOM) und dem Digitalen Geländemodell (DGM) berechnet:   $CHM = DOM - DGM$.",
        "Trees": "Abwandlung des CHM, welche nur Baumhöhen über 1 Meter berücksichtigt und in trees (1) und no-trees (0) klassifiziert.",
    }

    return index_info


def load_map_layer(layers, country_code):
    Map = geemap.foliumap.Map()
    Map.add_basemap("SATELLITE")
    for key, value in st.session_state.layer.items():
        if key in layers:
            Map.addLayer(
                value.clip(st.session_state.country_aoi),
                get_layer_visualization_params(key),
                key,
                opacity=0.5,
            )
            Map.addLayer(
                ee.Image()
                .byte()
                .paint(
                    featureCollection=st.session_state.country_aoi, color=1, width=2
                ),
                {"palette": "FF0000"},
                "Country AOI",
                opacity=1,
            )
            Map.centerObject(st.session_state.country_aoi, 6)

    return Map

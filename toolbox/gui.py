import streamlit as st
from pathlib import Path
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
import ee
import tempfile
from streamlit_utils import *


# @st.cache_data
def load_map_layer(layers, country_code):
    Map = geemap.Map()
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
        .paint(featureCollection=st.session_state.country_aoi, color=1, width=2),
        {"palette": "FF0000"},
        "Country AOI",
        opacity=1,
    )
    Map.centerObject(st.session_state.country_aoi, 6)

    return Map


st.set_page_config(layout="wide", page_title="SDM Playground", page_icon=":bird:")

if "background_gdf" not in st.session_state:
    st.session_state.background_gdf = load_background_data()

initialize_gee()

st.title("SDM Plyaground")
st.write("This is a simple SDM visualizer using Streamlit.")

sdm_tab, what_if, stats_tab, extract_tab, faq_tab = st.tabs(
    ["SDM", "What if ...?", "Stats overview", "Extract", "FAQ"]
)

with sdm_tab:
    st.header("Species Distribution Modeling (SDM)")
    sdm_layer_col, index_info_col = st.columns([3, 1])
    with sdm_layer_col:
        with st.expander("Layer Settings", expanded=True):

            def create_layer_form():
                with st.form("layer_form", border=False):
                    year_col, species_col, country_col, county_col = st.columns(
                        4, gap="medium"
                    )
                    with year_col:
                        st.number_input(
                            "Select Year",
                            min_value=2015,
                            max_value=2024,
                            value=2024,
                            step=1,
                            key="year_input",
                        )
                        if (
                            "layer" not in st.session_state
                            or st.session_state.layer is None
                        ):
                            st.session_state.layer = get_layer_information(
                                st.session_state.year_input
                            )
                    with species_col:
                        st.text_input(
                            "Species Name", value="Lagopus muta", key="species_input"
                        )

                    with country_col:
                        st.text_input(
                            "Country Code", value="AT", max_chars=2, key="country_input"
                        )
                    with county_col:
                        county = st.text_input(
                            "County Name (Prediction Layer)",
                            value="Tirol",
                            key="county_input",
                        )
                    st.session_state.country_aoi, st.session_state.county_aoi = (
                        get_aoi_from_nuts(
                            st.session_state.country_input,
                            st.session_state.county_input,
                        )
                    )
                    if (
                        st.session_state.species_input
                        and st.session_state.country_input
                    ):
                        st.session_state.species_gdf = get_species_data(
                            st.session_state.species_input,
                            st.session_state.country_input,
                        )

                    st.multiselect(
                        "Select Layer to display",
                        options=list(st.session_state.layer.keys()),
                        default=["landcover", "GHMI", "CHM"],
                        key="layers_select",
                    )
                    if st.form_submit_button("Update Layers"):
                        Map = load_map_layer(
                            st.session_state.layers_select,
                            st.session_state.country_input,
                        )

            create_layer_form()

        with st.expander("SDM Settings", expanded=False):

            @st.fragment
            def create_sdm_form():
                with st.form("sdm_form", border=False):
                    model_col, tree_col, tree_depth_col, train_size_col = st.columns(
                        4, gap="medium"
                    )
                    with model_col:
                        st.selectbox(
                            "Select Model",
                            options=["Random Forest", "Maxent", "Embedding"],
                            index=0,
                            key="model_input",
                        )
                    with tree_col:
                        st.number_input(
                            "Number of Trees (CART)",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10,
                            key="n_trees_input",
                        )
                    with tree_depth_col:
                        st.number_input(
                            "Max Tree Depth (CART)",
                            min_value=1,
                            max_value=50,
                            value=3,
                            step=1,
                            key="tree_depth_input",
                        )
                    with train_size_col:
                        st.number_input(
                            "Train Size (%)",
                            min_value=50,
                            max_value=90,
                            value=75,
                            step=5,
                            key="train_size_input",
                        )

                    st.multiselect(
                        "Select Environmental Features",
                        options=list(st.session_state.layer.keys()),
                        default=["landcover", "NDVI", "CHM", "b1", "b12"],
                        key="features_select",
                    )

                    if st.form_submit_button("Run SDM"):
                        if "presence_gdf" in st.session_state:
                            try:
                                st.session_state.presence_gdf = (
                                    st.session_state.presence_gdf[
                                        st.session_state.features_select + ["geometry"]
                                    ]
                                )
                            except:
                                (
                                    st.session_state.presence_gdf,
                                    st.session_state.predictors,
                                ) = get_species_features(
                                    _species_gdf=st.session_state.species_gdf,
                                    features=list(st.session_state.features_select),
                                    _layer=st.session_state.layer,
                                )
                        else:
                            (
                                st.session_state.presence_gdf,
                                st.session_state.predictors,
                            ) = get_species_features(
                                _species_gdf=st.session_state.species_gdf,
                                features=list(st.session_state.features_select),
                                _layer=st.session_state.layer,
                            )

                        with st.spinner("Running SDM..."):
                            (
                                st.session_state.model,
                                st.session_state.results_df,
                                st.session_state.ml_gdf,
                            ) = compute_sdm(
                                presence=st.session_state.presence_gdf,
                                background=st.session_state.background_gdf,
                                features=list(st.session_state.features_select),
                                model_type=st.session_state.model_input,
                                n_trees=st.session_state.n_trees_input,
                                tree_depth=st.session_state.tree_depth_input,
                                train_size=st.session_state.train_size_input / 100,
                            )

                            st.success(
                                "SDM run completed. Results are displayed on the map below."
                            )
                            try:
                                st.write("Model Performance Metrics:")
                                st.table(st.session_state.results_df)
                                if (
                                    rocauc_mean := st.session_state.results_df[
                                        "roc_auc"
                                    ].mean()
                                ) > 0.7:
                                    st.success("Mean ROC-AUC: {}".format(rocauc_mean))
                                else:
                                    st.error(
                                        "Mean ROC-AUC: {} < 0.7".format(rocauc_mean)
                                    )
                            except:
                                pass

            create_sdm_form()
    with index_info_col:

        @st.fragment
        def create_index_info():
            st.subheader("What is this Index about?")
            st.session_state.index_info = get_index_info()
            st.selectbox(
                "Choose an index to learn more",
                options=list(st.session_state.layer.keys()),
                key="index_select",
                index=0,
            )
            if "index_select" in st.session_state:
                st.markdown(
                    st.session_state.index_info.get(
                        st.session_state.index_select,
                        "No information available for this index.",
                    )
                )

        create_index_info()

    with st.expander("Map View", expanded=True):

        @st.fragment
        def create_map():
            if "country_input" in st.session_state:
                Map = load_map_layer(
                    st.session_state.layers_select, st.session_state.country_input
                )

                if (
                    st.button(f"Classify {st.session_state.county_input} Layer")
                    and "model" in st.session_state
                ):
                    with st.spinner("Classifying image..."):
                        if "model" in st.session_state:
                            st.write("✅ Model found.")
                        else:
                            st.error("❌ Run the SDM first to see the prediction.")
                        st.write(
                            "Sample presence size: {}, Sample pseudo-absence: {}".format(
                                st.session_state.ml_gdf[
                                    st.session_state.ml_gdf.PresAbs == 1
                                ].shape[0],
                                st.session_state.ml_gdf[
                                    st.session_state.ml_gdf.PresAbs == 0
                                ].shape[0],
                            )
                        )
                        st.session_state.classified_img_pr = classify_image_aoi(
                            image=st.session_state.predictors,
                            aoi=st.session_state.county_aoi,
                            ml_gdf=st.session_state.ml_gdf,
                            model=st.session_state.model,
                            features=list(st.session_state.features_select),
                        )
                        st.success("Image classified.")
                if "classified_img_pr" in st.session_state:
                    Map.addLayer(
                        st.session_state.classified_img_pr,
                        {
                            "min": 0,
                            "max": 1,
                            "palette": geemap.colormaps.palettes.viridis_r,
                        },
                        "Habitat suitability",
                    )
                    Map.add_colorbar(
                        {
                            "min": 0,
                            "max": 1,
                            "palette": geemap.colormaps.palettes.viridis_r,
                        },
                        label="Habitat suitability",
                        orientation="vertical",
                        position="bottomright",
                        layer_name="Habitat suitability",
                    )
                    Map.addLayer(
                        geemap.gdf_to_ee(
                            st.session_state.ml_gdf[
                                st.session_state.ml_gdf.PresAbs == 0
                            ]
                        ),
                        {"color": "orange"},
                        "Used background data points",
                        shown=False,
                    )
                    # if st.button(label="Download Prediction GeoTIFF"):
                    #     geemap.ee_export_image(
                    #         ee_object=st.session_state.classified_img_pr,
                    #         region=st.session_state.classified_img_pr.geometry(),
                    #         scale=90,
                    #         filename=Path.home() / f'Downloads/SDM_Prediction_{st.session_state.species_input}_{st.session_state.county_input}_{st.session_state.year_input}.tif'
                    #     )
                if st.session_state.species_gdf is not None:
                    Map.addLayer(
                        geemap.gdf_to_ee(st.session_state.species_gdf),
                        {"color": "red"},
                        f"Species Observations {st.session_state.species_input}",
                        shown=True,
                    )
                    Map.addLayer(
                        geemap.gdf_to_ee(st.session_state.background_gdf),
                        {"color": "blue"},
                        "Background data",
                        shown=False,
                    )

                Map.to_streamlit()

        create_map()


with what_if:
    st.header("What if ...?")
    st.write(
        "Explore hypothetical scenarios by modifying environmental variables and observing their impact on species distribution predictions."
    )
    st.write("This section is under development.")
    if "predictors" not in st.session_state:
        st.error("Run the SDM first to use the What-If analysis.")
    modify_features = st.multiselect(
        "Choose a feature to modify", options=st.session_state.features_select
    )
    feature_col, value_col = st.columns(2, gap="medium")
    with feature_col:
        if modify_features:
            for modify_feature in modify_features:
                st.number_input(
                    f"Set new value for {modify_feature}",
                    value=0.0,
                    key=f"modify_value_{modify_feature}",
                )
    with st.form("whatif_form", border=False):
        # Fill each column with content

        if st.form_submit_button("Run What-If Analysis"):
            st.session_state.future_predictors = st.session_state.predictors
            for modify_feature in modify_features:
                modify_value = st.session_state[f"modify_value_{modify_feature}"]
                st.write(f"Modifying {modify_feature} by adding {modify_value}.")
                future_value = (
                    st.session_state.predictors.select(modify_feature)
                    .add(ee.Image.constant(modify_value))
                    .rename(modify_feature)
                )
                temp = st.session_state.features_select.copy()
                temp.remove(modify_feature)
                st.session_state.future_predictors = (
                    st.session_state.future_predictors.select(temp).addBands(
                        future_value
                    )
                )

            st.session_state.classified_img__ftr_pr = classify_image_aoi(
                image=st.session_state.future_predictors,
                aoi=st.session_state.county_aoi,
                ml_gdf=st.session_state.ml_gdf,
                model=st.session_state.model,
                features=st.session_state.features_select,
            )
            st.session_state.change_layer = st.session_state.classified_img_pr.divide(
                st.session_state.classified_img__ftr_pr
            ).rename("Change in Habitat Suitability")
            st.success("What-If analysis completed.")
            if "classified_img__ftr_pr" in st.session_state:
                Map = geemap.foliumap.Map()
                Map.add_basemap("SATELLITE")
                Map.addLayer(
                    st.session_state.classified_img__ftr_pr,
                    {
                        "min": 0,
                        "max": 1,
                        "palette": geemap.colormaps.palettes.viridis_r,
                    },
                    "Future Habitat suitability",
                )
                Map.add_colorbar(
                    {
                        "min": 0,
                        "max": 1,
                        "palette": geemap.colormaps.palettes.viridis_r,
                    },
                    label="Habitat suitability",
                    orientation="vertical",
                    position="bottomright",
                    layer_name="Habitat suitability",
                )
                Map.addLayer(
                    st.session_state.classified_img_pr,
                    {
                        "min": 0,
                        "max": 1,
                        "palette": geemap.colormaps.palettes.viridis_r,
                    },
                    "Habitat suitability",
                )

                Map.addLayer(
                    st.session_state.change_layer,
                    {"min": 0.5, "max": 1.5, "palette": ["green", "yellow", "red"]},
                    "Change in Habitat suitability",
                )
                Map.add_colorbar(
                    {"min": 0.5, "max": 1.5, "palette": ["green", "yellow", "red"]},
                    label="Change in Habitat suitability",
                    orientation="vertical",
                    position="bottomright",
                    layer_name="Change in Habitat suitability",
                )
                Map.addLayer(
                    geemap.gdf_to_ee(st.session_state.species_gdf),
                    {"color": "red"},
                    f"Species Observations {st.session_state.species_input}",
                    shown=True,
                )
                Map.to_streamlit()

with stats_tab:
    with st.expander("Stats and Visualizations"):
        st.write("Statistical summaries and visualizations will be displayed here.")
        if (
            st.session_state.background_gdf is not None
            and st.session_state.features_select
        ):
            # st.write(st.session_state.background_gdf.drop('geometry', axis=1).corr())
            st.write(
                "Hierarchical Clustering of Features (left) and Correlation matrix (right):"
            )
            fig, selected_features = plot_hier_clustering(
                st.session_state.background_gdf
            )
            st.pyplot(fig=fig)
            st.write(f"Selected features: {list(selected_features)}")


with faq_tab:
    st.header("Frequently Asked Questions (FAQ)")


if __name__ == "__main__":
    pass

"""Step 5 — Results & Export."""

from __future__ import annotations

from typing import Callable

import branca.colormap as cm
import ee
import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geemap
from nicegui import context as nicegui_context, ui

from app.map_server import make_iframe, set_iframe_map
from app.state import AppState

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_VIS_PARAMS = {
    "min": 0,
    "max": 1,
    "palette": [
        "#313695",
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#ffffbf",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#a50026",
    ],
}

_OCCURRENCE_LEGEND = """<div style="position:fixed;bottom:20px;right:20px;z-index:999;
    background:white;padding:8px 12px;border-radius:6px;
    box-shadow:0 2px 6px rgba(0,0,0,.3);font:12px Arial,sans-serif;">
  <b>Legend</b><br>
  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
    background:#e74c3c;margin-right:5px;"></span>Occurrence
</div>"""


def _build_occurrence_map_html(state: AppState) -> str:
    """Return folium HTML with occurrence points only (no GEE layer), or '' on failure."""
    if state.species_gdf is None or state.species_gdf.empty:
        return ""
    try:
        bounds = state.species_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        for _, row in state.species_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color="#e74c3c",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.7,
                weight=1,
            ).add_to(fmap)
        fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        fmap.get_root().html.add_child(folium.Element(_OCCURRENCE_LEGEND))
        return fmap.get_root().render()
    except Exception:
        return ""


async def _build_map_html_async(state: AppState) -> str:
    """Return folium HTML for the suitability map; getMapId runs off the event loop."""
    import asyncio

    loop = asyncio.get_event_loop()

    if state.classified_img is None:
        return _build_occurrence_map_html(state)

    # Derive centre from presence points, fall back to world view.
    bounds = None
    if state.species_gdf is not None and not state.species_gdf.empty:
        bounds = state.species_gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    else:
        center = [20, 0]

    try:
        # Blocking GEE network call — must not run on the event loop thread.
        classified_img = state.classified_img
        map_id = await loop.run_in_executor(
            None, lambda: classified_img.getMapId(_VIS_PARAMS)
        )
        tile_url = map_id["tile_fetcher"].url_format

        fmap = folium.Map(location=center, zoom_start=6)
        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine",
            name="Habitat Suitability",
            overlay=True,
        ).add_to(fmap)
        colormap = cm.LinearColormap(
            colors=_VIS_PARAMS["palette"],
            vmin=0,
            vmax=1,
            caption="Habitat Suitability",
        )
        colormap.add_to(fmap)
        fmap.get_root().html.add_child(folium.Element(_OCCURRENCE_LEGEND))

        if bounds is not None:
            for _, row in state.species_gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color="#e74c3c",
                    fill=True,
                    fill_color="#e74c3c",
                    fill_opacity=0.7,
                    weight=1,
                ).add_to(fmap)
            fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        return fmap.get_root().render()
    except Exception:
        return _build_occurrence_map_html(state)


_METRIC_COLS = {"roc_auc", "overall_accuracy"}


def _has_feature_importances(state: AppState) -> bool:
    """Return True when feature importance columns are present in results_df."""
    if state.model_type not in ("rf",) or state.results_df is None:
        return False
    feature_cols = [c for c in state.results_df.columns if c not in _METRIC_COLS]
    return len(feature_cols) > 0


def _make_feature_importance_fig(state: AppState) -> plt.Figure:
    """Create a horizontal bar chart of feature importances, sorted descending."""
    feature_cols = [c for c in state.results_df.columns if c not in _METRIC_COLS]
    means = state.results_df[feature_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, max(3, len(feature_cols) * 0.4)))
    colors = plt.cm.RdYlBu_r(  # type: ignore[attr-defined]
        [v / means.max() for v in means.values]
    )
    ax.barh(means.index, means.values, color=colors)
    ax.set_xlabel("Mean Importance")
    ax.set_title("Feature Importances")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(state: AppState, on_back: Callable) -> None:
    """Render Step 5: Results & Export.

    Parameters
    ----------
    state:
        Shared ``AppState`` instance; mutations are reflected app-wide.
    on_back:
        Callable invoked when the user clicks the "← Back" button.
    """
    import asyncio

    if state.classified_img is None:
        ui.notification(
            "Results not ready — please run the model first.", type="warning"
        )
        on_back()
        return

    _client = nicegui_context.client

    # Mutable refs for widgets that inner closures need to reach.
    _map_iframe_ref: list[ui.element] = []
    _map_loading_ref: list[ui.label] = []
    _export_status_ref: list[ui.label] = []
    _whatif_offset_inputs: dict[str, list[ui.number]] = {}
    _reclassify_status_ref: list[ui.label] = []
    _reclassify_progress_ref: list[ui.linear_progress] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_map(html: str) -> None:
        """Push new folium HTML into the persistent iframe element."""
        if _map_loading_ref:
            _map_loading_ref[0].set_visibility(False)
        if _map_iframe_ref and html:
            set_iframe_map(_map_iframe_ref[0], html)

    def _export() -> None:
        """Start a GEE export task to Google Drive (non-blocking: just submits the task)."""
        if state.classified_img is None:
            if _export_status_ref:
                _export_status_ref[0].set_text(
                    "No classified image available to export."
                )
                _export_status_ref[0].set_visibility(True)
            return

        async def _run() -> None:
            loop = asyncio.get_event_loop()
            try:

                def _start_export():
                    from toolbox.utils import get_aoi_from_nuts

                    country_aoi, county_aoi = get_aoi_from_nuts(
                        country_code=state.country_code,
                        county_name=state.county_name or None,
                    )
                    aoi = county_aoi if county_aoi is not None else country_aoi
                    clipped_img = state.classified_img.clip(aoi)

                    task = ee.batch.Export.image.toDrive(
                        image=clipped_img,
                        description="SDM_Prediction",
                        fileFormat="GeoTIFF",
                        scale=90,
                        maxPixels=1e13,
                    )
                    task.start()
                    return "Export started. Check Google Earth Engine Tasks tab."

                msg = await loop.run_in_executor(None, _start_export)
            except Exception as exc:  # noqa: BLE001
                msg = f"Export failed: {exc}"
            with _client:
                if _export_status_ref:
                    _export_status_ref[0].set_text(msg)
                    _export_status_ref[0].set_visibility(True)

        asyncio.ensure_future(_run())

    def _reclassify() -> None:
        """Apply what-if offsets and re-classify using the stored GEE classifier."""
        if state.model is None or state.model_type == "embedding":
            if _reclassify_status_ref:
                _reclassify_status_ref[0].set_text(
                    "What-If is not available for the Embedding model."
                )
                _reclassify_status_ref[0].set_visibility(True)
            return

        if state.layer_stack is None or not state.selected_layers:
            if _reclassify_status_ref:
                _reclassify_status_ref[0].set_text(
                    "Layer stack not available — cannot re-classify."
                )
                _reclassify_status_ref[0].set_visibility(True)
            return

        if _reclassify_progress_ref:
            _reclassify_progress_ref[0].set_visibility(True)
            _reclassify_progress_ref[0].value = 0.0
        if _reclassify_status_ref:
            _reclassify_status_ref[0].set_text("Re-classifying...")
            _reclassify_status_ref[0].set_visibility(True)

        async def _run() -> None:
            loop = asyncio.get_event_loop()

            for layer, refs in _whatif_offset_inputs.items():
                if refs:
                    try:
                        state.whatif_offsets[layer] = float(refs[0].value or 0.0)
                    except (TypeError, ValueError):
                        state.whatif_offsets[layer] = 0.0

            try:
                from toolbox.utils import get_aoi_from_nuts

                with _client:
                    if _reclassify_progress_ref:
                        _reclassify_progress_ref[0].value = 0.3

                def _do_reclassify():
                    modified_predictors = ee.Image.cat(
                        [
                            state.layer_stack[k].add(
                                ee.Image.constant(state.whatif_offsets.get(k, 0.0))
                            )
                            for k in state.selected_layers
                            if k in state.layer_stack
                        ]
                    )
                    country_aoi, county_aoi = get_aoi_from_nuts(
                        country_code=state.country_code,
                        county_name=state.county_name or None,
                    )
                    aoi = county_aoi if county_aoi is not None else country_aoi
                    if state.model_type == "maxent":
                        return (
                            modified_predictors.clip(aoi)
                            .classify(state.model)
                            .select("probability")
                        )
                    return modified_predictors.clip(aoi).classify(state.model)

                with _client:
                    if _reclassify_progress_ref:
                        _reclassify_progress_ref[0].value = 0.5

                new_img = await loop.run_in_executor(None, _do_reclassify)

                with _client:
                    if _reclassify_progress_ref:
                        _reclassify_progress_ref[0].value = 0.8

                state.classified_img = new_img
                html = await _build_map_html_async(state)
                with _client:
                    _refresh_map(html)

                    if _reclassify_progress_ref:
                        _reclassify_progress_ref[0].value = 1.0
                        _reclassify_progress_ref[0].set_visibility(False)
                    if _reclassify_status_ref:
                        _reclassify_status_ref[0].set_text("Re-classification complete.")

            except Exception as exc:  # noqa: BLE001
                with _client:
                    if _reclassify_progress_ref:
                        _reclassify_progress_ref[0].set_visibility(False)
                    if _reclassify_status_ref:
                        _reclassify_status_ref[0].set_text(
                            f"Re-classification failed: {exc}"
                        )

        asyncio.ensure_future(_run())

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-4xl mx-auto"):
        ui.label("Step 5 — Results & Export").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-6"):
            # 1. Folium map — tile URL fetched async so the event loop isn't blocked.
            loading_lbl = ui.label("Loading map…").classes(
                "text-sm text-gray-400 italic"
            )
            _map_loading_ref.append(loading_lbl)
            map_iframe = make_iframe(height="55vh")
            _map_iframe_ref.append(map_iframe)

            async def _load_initial_map() -> None:
                html = await _build_map_html_async(state)
                with _client:
                    _refresh_map(html)
                    if not html:
                        loading_lbl.set_text("No classified image available.")
                        loading_lbl.set_visibility(True)

            asyncio.ensure_future(_load_initial_map())

            # 2. Feature importance chart (RF only)
            if _has_feature_importances(state):
                ui.label("Feature Importances").classes(
                    "font-semibold text-sm text-gray-600"
                )
                fig = _make_feature_importance_fig(state)
                import io, base64

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                ui.html(
                    f'<img src="data:image/png;base64,{b64}" style="width:100%">'
                ).classes("w-full")
                plt.close(fig)

            # 3. What-If panel
            with ui.expansion("What-If Analysis").classes("w-full"):
                with ui.column().classes("w-full gap-3 pt-2"):
                    if state.selected_layers:
                        ui.label(
                            "Adjust layer offsets and re-classify to explore "
                            "habitat suitability under modified conditions."
                        ).classes("text-sm text-gray-600")

                        for layer in state.selected_layers:
                            current_offset = state.whatif_offsets.get(layer, 0.0)

                            def _make_offset_handler(lyr: str):
                                def _handler(e) -> None:
                                    try:
                                        state.whatif_offsets[lyr] = float(
                                            e.value or 0.0
                                        )
                                    except (TypeError, ValueError):
                                        state.whatif_offsets[lyr] = 0.0

                                return _handler

                            num = ui.number(
                                label=f"Offset: {layer}",
                                value=current_offset,
                                step=0.1,
                                on_change=_make_offset_handler(layer),
                            ).classes("w-full")
                            _whatif_offset_inputs[layer] = [num]
                    else:
                        ui.label(
                            "No layers selected — return to Step 3 to select layers."
                        ).classes("text-sm text-gray-400 italic")

                    reclassify_status = ui.label("").classes("text-sm text-amber-700")
                    reclassify_status.set_visibility(False)
                    _reclassify_status_ref.append(reclassify_status)

                    reclassify_progress = ui.linear_progress(value=0.0).classes(
                        "w-full"
                    )
                    reclassify_progress.set_visibility(False)
                    _reclassify_progress_ref.append(reclassify_progress)

                    ui.button("Re-classify", on_click=_reclassify).classes("mt-2")

            # 4. Export button + status
            with ui.row().classes("w-full items-center gap-4"):
                ui.button("Export to Google Drive", on_click=_export).classes(
                    "bg-blue-600 text-white"
                )
                export_status = ui.label("").classes("text-sm")
                export_status.set_visibility(False)
                _export_status_ref.append(export_status)

            # 5. Navigation — Back only (last step)
            with ui.row().classes("w-full mt-2"):
                ui.button("← Back", on_click=on_back)

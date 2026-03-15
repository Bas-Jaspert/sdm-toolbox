"""Step 5 — Results & Export."""

from __future__ import annotations

from typing import Callable

import ee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geemap
import geemap.foliumap as geemap_folium
from nicegui import ui

from app.map_server import make_iframe, set_iframe_map
from app.state import AppState

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_map_html(state: AppState) -> str:
    """Return standalone folium HTML for the habitat suitability map, or '' on failure."""
    import folium
    try:
        if state.classified_img is None:
            return ""
        Map = geemap_folium.Map()
        Map.addLayer(
            state.classified_img,
            {
                "min": 0, "max": 1,
                "palette": [
                    "#313695", "#4575b4", "#74add1", "#abd9e9",
                    "#ffffbf",
                    "#fee090", "#fdae61", "#f46d43", "#a50026",
                ],
            },
            "Habitat Suitability",
        )
        if state.species_gdf is not None and not state.species_gdf.empty:
            Map.addLayer(
                geemap.gdf_to_ee(state.species_gdf),
                {"color": "red"},
                "Presence Points",
            )
        return Map.get_root().render()
    except Exception:
        # Fallback: plain folium map with presence points only
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
                    radius=4, color="#e74c3c", fill=True,
                    fill_color="#e74c3c", fill_opacity=0.7, weight=1,
                ).add_to(fmap)
            fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            return fmap.get_root().render()
        except Exception:
            return ""


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

    # Mutable refs for widgets that inner closures need to reach.
    _map_iframe_ref: list[ui.element] = []
    _export_status_ref: list[ui.label] = []
    _whatif_offset_inputs: dict[str, list[ui.number]] = {}
    _reclassify_status_ref: list[ui.label] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_map(html: str) -> None:
        """Push new folium HTML into the persistent iframe element."""
        if _map_iframe_ref and html:
            set_iframe_map(_map_iframe_ref[0], html)

    def _export() -> None:
        """Start a GEE export task to Google Drive (non-blocking: just submits the task)."""
        if state.classified_img is None:
            if _export_status_ref:
                _export_status_ref[0].set_text("No classified image available to export.")
                _export_status_ref[0].set_visibility(True)
            return

        async def _run() -> None:
            import asyncio
            loop = asyncio.get_event_loop()
            try:
                def _start_export():
                    task = ee.batch.Export.image.toDrive(
                        image=state.classified_img,
                        description="SDM_Prediction",
                        fileFormat="GeoTIFF",
                        scale=90,
                    )
                    task.start()
                    return "Export started. Check Google Earth Engine Tasks tab."
                msg = await loop.run_in_executor(None, _start_export)
            except Exception as exc:  # noqa: BLE001
                msg = f"Export failed: {exc}"
            if _export_status_ref:
                _export_status_ref[0].set_text(msg)
                _export_status_ref[0].set_visibility(True)

        import asyncio
        asyncio.ensure_future(_run())

    def _reclassify() -> None:
        """Apply what-if offsets and re-classify (local/sklearn models only)."""
        # Check whether re-classification is feasible
        if state.model is None or isinstance(state.model, str):
            if _reclassify_status_ref:
                _reclassify_status_ref[0].set_text(
                    "What-If re-classification requires the local pipeline "
                    "(Deep Dive mode). For GEE and Embedding modes, use the notebook."
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

        if state.ml_gdf is None:
            if _reclassify_status_ref:
                _reclassify_status_ref[0].set_text(
                    "Training data not available — What-If requires Deep Dive mode."
                )
                _reclassify_status_ref[0].set_visibility(True)
            return

        async def _run() -> None:
            import asyncio
            loop = asyncio.get_event_loop()

            # Read current offset values from inputs
            for layer, refs in _whatif_offset_inputs.items():
                if refs:
                    try:
                        state.whatif_offsets[layer] = float(refs[0].value or 0.0)
                    except (TypeError, ValueError):
                        state.whatif_offsets[layer] = 0.0

            try:
                from toolbox.utils import classify_image_aoi, get_aoi_from_nuts

                def _do_reclassify():
                    modified_predictors = ee.Image.cat([
                        state.layer_stack[k].add(
                            ee.Image.constant(state.whatif_offsets.get(k, 0.0))
                        )
                        for k in state.selected_layers
                        if k in state.layer_stack
                    ])
                    country_aoi, county_aoi = get_aoi_from_nuts(
                        country_code=state.country_code,
                        county_name=state.county_name or None,
                    )
                    aoi = county_aoi if county_aoi is not None else country_aoi
                    return classify_image_aoi(
                        modified_predictors,
                        aoi,
                        state.ml_gdf,
                        state.model,
                        state.selected_layers,
                    )

                new_img = await loop.run_in_executor(None, _do_reclassify)
                state.classified_img = new_img
                html = _build_map_html(state)
                _refresh_map(html)
                if _reclassify_status_ref:
                    _reclassify_status_ref[0].set_text("Re-classification complete.")
                    _reclassify_status_ref[0].set_visibility(True)

            except Exception as exc:  # noqa: BLE001
                if _reclassify_status_ref:
                    _reclassify_status_ref[0].set_text(f"Re-classification failed: {exc}")
                    _reclassify_status_ref[0].set_visibility(True)

        import asyncio
        asyncio.ensure_future(_run())

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-4xl mx-auto"):
        ui.label("Step 5 — Results & Export").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-6"):

            # 1. Folium map
            map_iframe = make_iframe(height="520px")
            _map_iframe_ref.append(map_iframe)
            map_html = _build_map_html(state)
            if map_html:
                set_iframe_map(map_iframe, map_html)
            else:
                ui.label("No classified image available.").classes(
                    "text-gray-400 italic text-sm"
                )

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
                ui.html(f'<img src="data:image/png;base64,{b64}" style="width:100%">').classes("w-full")
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
                                        state.whatif_offsets[lyr] = float(e.value or 0.0)
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

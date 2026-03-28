"""Step 3 — Environmental Layers & Model configuration."""

from __future__ import annotations

import asyncio
from typing import Callable

from nicegui import context as nicegui_context, ui

from app.state import AppState
from app.services import gee_service
from app.services.layer_metadata import CATEGORY_NOTES, get_catalogue_by_category

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_OPTIONS: dict[str, str] = {
    "rf": "Random Forest",
    "maxent": "Maxent",
    "embedding": "Embedding",
}

_RESOLUTION_OPTIONS: dict[int, str] = {
    10: "10 m",
    30: "30 m",
    100: "100 m",
    300: "300 m",
    1000: "1000 m",
}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(state: AppState, on_next: Callable, on_back: Callable) -> None:
    """Render Step 3: Environmental Layers & Model.

    Parameters
    ----------
    state:
        Shared ``AppState`` instance; mutations are reflected app-wide.
    on_next:
        Callable invoked when the user clicks the "Next →" button.
    on_back:
        Callable invoked when the user clicks the "← Back" button.
    """

    _client = nicegui_context.client

    # Mutable container references so inner closures can reach widgets.
    _next_btn_ref: list[ui.button] = []
    _layer_select_ref: list[ui.select] = []
    _hyperparam_section_ref: list[ui.element] = []
    _gee_section_ref: list[ui.element] = []
    _embedding_info_ref: list[ui.element] = []
    _rf_only_ref: list[ui.element] = []
    _init_btn_ref: list[ui.button] = []
    _status_label_ref: list[ui.label] = []
    _custom_layer_status_ref: list[ui.label] = []
    _custom_asset_id_ref: list[ui.input] = []
    _custom_band_ref: list[ui.input] = []
    _custom_name_ref: list[ui.input] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_next_button() -> None:
        """Enable Next button based on layer selection or embedding mode."""
        if state.model_type == "embedding":
            enabled = True
        else:
            enabled = state.layer_stack is not None and len(state.selected_layers) > 0
        if _next_btn_ref:
            _next_btn_ref[0].set_enabled(enabled)

    def _refresh_hyperparam_section() -> None:
        """Show/hide hyperparameter section and RF-only fields based on model type."""
        if _hyperparam_section_ref:
            _hyperparam_section_ref[0].set_visibility(state.model_type != "embedding")
        if _rf_only_ref:
            _rf_only_ref[0].set_visibility(state.model_type == "rf")

    def _refresh_gee_section() -> None:
        """Show/hide the GEE layer section and embedding info based on model type."""
        if _gee_section_ref:
            _gee_section_ref[0].set_visibility(state.model_type != "embedding")
        if _embedding_info_ref:
            _embedding_info_ref[0].set_visibility(state.model_type == "embedding")

    def _on_layer_change(e) -> None:
        state.selected_layers = list(e.value) if e.value else []
        _refresh_next_button()

    def _on_model_change(e) -> None:
        state.model_type = e.value or "rf"
        _refresh_hyperparam_section()
        _refresh_gee_section()
        _refresh_next_button()

    def _on_n_trees_change(e) -> None:
        try:
            state.n_trees = int(e.value)
        except (TypeError, ValueError):
            pass

    def _on_max_depth_change(e) -> None:
        try:
            state.max_depth = int(e.value)
        except (TypeError, ValueError):
            pass

    def _on_train_size_change(e) -> None:
        try:
            state.train_size = int(e.value) / 100
        except (TypeError, ValueError):
            pass

    def _on_resolution_change(e) -> None:
        if e.value in _RESOLUTION_OPTIONS:
            state.resolution = e.value

    def _init_layers() -> None:
        """Trigger GEE layer initialisation in a thread executor."""

        if _status_label_ref:
            _status_label_ref[0].set_text("Initializing GEE layers…")
            _status_label_ref[0].set_visibility(True)

        if _init_btn_ref:
            _init_btn_ref[0].set_visibility(False)

        async def _run() -> None:
            loop = asyncio.get_event_loop()
            try:
                layer_stack = await loop.run_in_executor(
                    None,
                    lambda: gee_service.get_layer_information(state.year_start),
                )
                state.layer_stack = layer_stack
                layer_names = list(layer_stack.keys())

                with _client:
                    # Update the layer multi-select options and restore any
                    # previously selected layers that still exist in the new stack.
                    if _layer_select_ref:
                        _layer_select_ref[0].set_options(layer_names)
                        valid_selection = [
                            lyr for lyr in state.selected_layers if lyr in layer_names
                        ]
                        _layer_select_ref[0].value = valid_selection
                        state.selected_layers = valid_selection

                    if _status_label_ref:
                        _status_label_ref[0].set_text(
                            f"GEE initialized — {len(layer_names)} layers available."
                        )

            except Exception as exc:  # noqa: BLE001
                with _client:
                    if _status_label_ref:
                        _status_label_ref[0].set_text(f"Error: {exc}")
                    if _init_btn_ref:
                        _init_btn_ref[0].set_visibility(True)
            finally:
                with _client:
                    _refresh_next_button()

        asyncio.ensure_future(_run())

    def _add_custom_layer() -> None:
        """Load a user-supplied GEE asset and add it to the layer stack."""
        asset_id = _custom_asset_id_ref[0].value.strip() if _custom_asset_id_ref else ""
        band = _custom_band_ref[0].value.strip() if _custom_band_ref else ""
        name = _custom_name_ref[0].value.strip() if _custom_name_ref else ""

        if not asset_id:
            if _custom_layer_status_ref:
                _custom_layer_status_ref[0].set_text("Asset ID is required.")
                _custom_layer_status_ref[0].set_visibility(True)
            return
        if not name:
            if _custom_layer_status_ref:
                _custom_layer_status_ref[0].set_text("Display name is required.")
                _custom_layer_status_ref[0].set_visibility(True)
            return

        if _custom_layer_status_ref:
            _custom_layer_status_ref[0].set_text("Loading layer…")
            _custom_layer_status_ref[0].set_visibility(True)

        async def _run() -> None:
            loop = asyncio.get_event_loop()
            try:
                img = await loop.run_in_executor(
                    None,
                    lambda: gee_service.load_custom_layer(
                        asset_id, band or None, name
                    ),
                )
                if state.layer_stack is None:
                    state.layer_stack = {}
                state.layer_stack[name] = img

                layer_names = list(state.layer_stack.keys())
                new_selection = list(state.selected_layers) + [name]

                with _client:
                    if _layer_select_ref:
                        _layer_select_ref[0].set_options(layer_names)
                        _layer_select_ref[0].set_value(new_selection)
                    state.selected_layers = new_selection
                    if _custom_layer_status_ref:
                        _custom_layer_status_ref[0].set_text(
                            f"Layer '{name}' added successfully."
                        )
                    # Clear inputs for next use
                    if _custom_asset_id_ref:
                        _custom_asset_id_ref[0].set_value("")
                    if _custom_band_ref:
                        _custom_band_ref[0].set_value("")
                    if _custom_name_ref:
                        _custom_name_ref[0].set_value("")
                    _refresh_next_button()

            except Exception as exc:
                with _client:
                    if _custom_layer_status_ref:
                        _custom_layer_status_ref[0].set_text(f"Error: {exc}")

        asyncio.ensure_future(_run())

    # ------------------------------------------------------------------
    # Initial layer options — use existing stack if already populated
    # ------------------------------------------------------------------

    _initial_layer_options: list[str] = (
        list(state.layer_stack.keys()) if state.layer_stack is not None else []
    )

    # ------------------------------------------------------------------
    # Layer catalogue dialog
    # ------------------------------------------------------------------

    def _open_layer_catalogue() -> None:
        catalogue = get_catalogue_by_category()
        with ui.dialog() as dlg, ui.card().classes(
            "w-full max-w-2xl max-h-screen overflow-y-auto p-6"
        ):
            ui.label("Environmental Layer Catalogue").classes(
                "text-xl font-bold mb-4"
            )
            for category, items in catalogue.items():
                ui.label(category).classes(
                    "text-base font-semibold mt-4 mb-1 text-primary"
                )
                ui.separator()
                for layer_name, meta in items:
                    with ui.column().classes("gap-0 my-1"):
                        ui.label(layer_name).classes("font-mono font-semibold text-sm")
                        ui.label(meta.description).classes("text-sm")
                        ui.label(
                            f"Units: {meta.units}  ·  Source: {meta.data_source}"
                        ).classes("text-xs text-gray-500")
                note = CATEGORY_NOTES.get(category)
                if note:
                    ui.label(note).classes("text-xs text-gray-400 italic mt-1")
            ui.button("Close", on_click=dlg.close).classes("mt-4 w-full")
        dlg.open()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-xl mx-auto"):
        ui.label("Step 3 — Layers & Model").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-4"):
            # 1. GEE layer section (hidden for embedding model)
            with ui.column().classes("w-full gap-2") as gee_section:
                # 1a. Layer multi-select + catalogue info button
                with ui.row().classes("w-full items-end gap-2"):
                    layer_select = ui.select(
                        label="Environmental Layers",
                        multiple=True,
                        options=_initial_layer_options,
                        value=state.selected_layers,
                        on_change=_on_layer_change,
                    ).classes("flex-1").props("use-chips")
                    _layer_select_ref.append(layer_select)

                    ui.button(
                        icon="info",
                        on_click=_open_layer_catalogue,
                    ).props("flat round").tooltip("Environmental layer catalogue")

                # 1b. Status label + hidden retry button (shown only on error)
                status_label = ui.label("Initializing GEE layers…").classes("text-sm")
                status_label.set_visibility(True)
                _status_label_ref.append(status_label)

                retry_btn = ui.button(
                    "Retry",
                    on_click=_init_layers,
                ).classes("w-full")
                retry_btn.set_visibility(False)
                _init_btn_ref.append(retry_btn)

                # Pre-populate status if stack is already loaded
                if state.layer_stack is not None:
                    status_label.set_text(
                        f"GEE initialized — {len(_initial_layer_options)} layers available."
                    )

                # 1c. Custom GEE layer panel
                with ui.expansion("Add custom layer", icon="add_circle").classes(
                    "w-full"
                ):
                    with ui.column().classes("w-full gap-2"):
                        ui.label(
                            "Add any GEE Image asset as an extra predictor layer. "
                            "Custom layers are static — they use the image you provide "
                            "for all presence points regardless of observation year."
                        ).classes("text-xs text-gray-500")

                        custom_asset_id = ui.input(
                            label="Asset ID",
                            placeholder="projects/my-project/assets/dem",
                        ).classes("w-full")
                        _custom_asset_id_ref.append(custom_asset_id)

                        custom_band = ui.input(
                            label="Band (optional — leave blank for first band)",
                            placeholder="b1",
                        ).classes("w-full")
                        _custom_band_ref.append(custom_band)

                        custom_name = ui.input(
                            label="Display name",
                            placeholder="My DEM",
                        ).classes("w-full")
                        _custom_name_ref.append(custom_name)

                        ui.button(
                            "Add layer",
                            on_click=_add_custom_layer,
                        ).classes("w-full")

                        custom_layer_status = ui.label("").classes("text-sm")
                        custom_layer_status.set_visibility(False)
                        _custom_layer_status_ref.append(custom_layer_status)

            _gee_section_ref.append(gee_section)

            # 2. Model type select
            ui.select(
                label="Model",
                options=_MODEL_OPTIONS,
                value=state.model_type,
                on_change=_on_model_change,
            ).classes("w-full")

            # 2b. Embedding info (shown only for embedding mode)
            with ui.column() as embedding_info:
                ui.label(
                    f"Embedding mode: uses Google Satellite Embedding V1 "
                    f"(year {state.year_start}-{state.year_end}, will use closest available)"
                ).classes("text-sm text-blue-600")
            _embedding_info_ref.append(embedding_info)

            # 3. Hyperparameters section (hidden for "embedding")
            with ui.column().classes(
                "w-full gap-2 pl-4 border-l-2 border-gray-300"
            ) as hyperparam_section:
                ui.label("Hyperparameters").classes(
                    "font-semibold text-sm text-gray-600"
                )

                # RF-only fields (hidden for Maxent and Embedding)
                with ui.column().classes("w-full gap-2") as rf_only:
                    ui.number(
                        label="Number of Trees",
                        value=state.n_trees,
                        min=10,
                        max=500,
                        step=10,
                        on_change=_on_n_trees_change,
                    ).classes("w-full")

                    ui.number(
                        label="Max Tree Depth",
                        value=state.max_depth,
                        min=1,
                        max=50,
                        step=1,
                        on_change=_on_max_depth_change,
                    ).classes("w-full")

                _rf_only_ref.append(rf_only)

                ui.number(
                    label="Train Size (%)",
                    value=int(state.train_size * 100),
                    min=50,
                    max=90,
                    step=5,
                    on_change=_on_train_size_change,
                ).classes("w-full")

                ui.select(
                    label="Resolution (m)",
                    options=_RESOLUTION_OPTIONS,
                    value=state.resolution,
                    on_change=_on_resolution_change,
                ).classes("w-full")

            _hyperparam_section_ref.append(hyperparam_section)

            # Reflect initial visibility
            _refresh_hyperparam_section()
            _refresh_gee_section()

            # 5. Navigation buttons
            with ui.row().classes("w-full justify-between mt-2"):
                ui.button("← Back", on_click=on_back)

                next_btn = ui.button(
                    "Next →",
                    on_click=on_next,
                )
                _next_btn_ref.append(next_btn)

            # Reflect initial state
            _refresh_next_button()

    # Auto-initialize GEE layers when entering the step for the first time.
    # Skip for embedding — it never uses GEE layers.
    if state.model_type != "embedding" and state.layer_stack is None:
        _init_layers()

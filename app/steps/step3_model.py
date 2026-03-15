"""Step 3 — Environmental Layers & Model configuration."""

from __future__ import annotations

import asyncio
from typing import Callable

from nicegui import ui

from app.state import AppState
from app.services import gee_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_OPTIONS: dict[str, str] = {
    "rf": "Random Forest",
    "maxent": "Maxent",
    "embedding": "Embedding",
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

    # Mutable container references so inner closures can reach widgets.
    _next_btn_ref: list[ui.button] = []
    _layer_select_ref: list[ui.select] = []
    _hyperparam_section_ref: list[ui.element] = []
    _init_btn_ref: list[ui.button] = []
    _status_label_ref: list[ui.label] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_next_button() -> None:
        """Enable/disable Next button based on layer selection and stack."""
        enabled = (
            state.layer_stack is not None
            and len(state.selected_layers) > 0
        )
        if _next_btn_ref:
            _next_btn_ref[0].set_enabled(enabled)

    def _refresh_hyperparam_section() -> None:
        """Show/hide hyperparameter section based on model type."""
        if _hyperparam_section_ref:
            _hyperparam_section_ref[0].set_visibility(
                state.model_type != "embedding"
            )

    def _on_layer_change(e) -> None:
        state.selected_layers = list(e.value) if e.value else []
        _refresh_next_button()

    def _on_model_change(e) -> None:
        state.model_type = e.value or "rf"
        _refresh_hyperparam_section()

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

    def _init_layers() -> None:
        """Trigger GEE layer initialisation in a thread executor."""

        if _status_label_ref:
            _status_label_ref[0].set_text("Initializing GEE layers…")
            _status_label_ref[0].set_visibility(True)

        if _init_btn_ref:
            _init_btn_ref[0].set_enabled(False)

        async def _run() -> None:
            loop = asyncio.get_event_loop()
            try:
                layer_stack = await loop.run_in_executor(
                    None,
                    lambda: gee_service.get_layer_information(state.year),
                )
                state.layer_stack = layer_stack
                layer_names = list(layer_stack.keys())

                # Update the layer multi-select options and restore any
                # previously selected layers that still exist in the new stack.
                if _layer_select_ref:
                    _layer_select_ref[0].set_options(layer_names)
                    valid_selection = [
                        lyr for lyr in state.selected_layers
                        if lyr in layer_names
                    ]
                    _layer_select_ref[0].value = valid_selection
                    state.selected_layers = valid_selection

                if _status_label_ref:
                    _status_label_ref[0].set_text(
                        f"GEE initialized — {len(layer_names)} layers available."
                    )

            except Exception as exc:  # noqa: BLE001
                if _status_label_ref:
                    _status_label_ref[0].set_text(f"Error: {exc}")
            finally:
                if _init_btn_ref:
                    _init_btn_ref[0].set_enabled(True)
                _refresh_next_button()

        asyncio.ensure_future(_run())

    # ------------------------------------------------------------------
    # Initial layer options — use existing stack if already populated
    # ------------------------------------------------------------------

    _initial_layer_options: list[str] = (
        list(state.layer_stack.keys()) if state.layer_stack is not None else []
    )

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-xl mx-auto"):
        ui.label("Step 3 — Layers & Model").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-4"):

            # 1. Layer multi-select
            layer_select = ui.select(
                label="Environmental Layers",
                multiple=True,
                options=_initial_layer_options,
                value=state.selected_layers,
                on_change=_on_layer_change,
            ).classes("w-full")
            _layer_select_ref.append(layer_select)

            # 2. Initialize GEE Layers button + status
            init_btn = ui.button(
                "Initialize GEE Layers",
                on_click=_init_layers,
            ).classes("w-full")
            _init_btn_ref.append(init_btn)

            status_label = ui.label("").classes("text-sm")
            status_label.set_visibility(False)
            _status_label_ref.append(status_label)

            # Pre-populate status if stack is already loaded
            if state.layer_stack is not None:
                status_label.set_text(
                    f"GEE initialized — {len(_initial_layer_options)} layers available."
                )
                status_label.set_visibility(True)

            # 3. Model type select
            ui.select(
                label="Model",
                options=_MODEL_OPTIONS,
                value=state.model_type,
                on_change=_on_model_change,
            ).classes("w-full")

            # 4. Hyperparameters section (hidden for "embedding")
            with ui.column().classes(
                "w-full gap-2 pl-4 border-l-2 border-gray-300"
            ) as hyperparam_section:
                ui.label("Hyperparameters").classes(
                    "font-semibold text-sm text-gray-600"
                )

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

                ui.number(
                    label="Train Size (%)",
                    value=int(state.train_size * 100),
                    min=50,
                    max=90,
                    step=5,
                    on_change=_on_train_size_change,
                ).classes("w-full")

            _hyperparam_section_ref.append(hyperparam_section)

            # Reflect initial visibility
            _refresh_hyperparam_section()

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

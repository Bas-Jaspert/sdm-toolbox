"""Step 4 — Run SDM."""

from __future__ import annotations

import asyncio
from typing import Callable

from nicegui import ui

from app.state import AppState
from app.services import sdm_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIPELINE_LABELS: dict[str, str] = {
    "embedding": "Embedding dot-product pipeline",
    "gee": "GEE-native pipeline (server-side, fast)",
    "local": "Local sklearn pipeline (Deep Dive, downloads features)",
}

_GEE_STEPS: list[str] = [
    "Loading AOI...",
    "Sampling features...",
    "Training classifier...",
    "Classifying AOI...",
]

_LOCAL_STEPS: list[str] = [
    "Loading AOI...",
    "Extracting features...",
    "Training local model...",
    "Classifying AOI...",
]

_EMBEDDING_STEPS: list[str] = [
    "Loading embedding mosaic...",
    "Sampling presence embeddings...",
    "Computing mean vector...",
    "Dot product similarity...",
    "Evaluating AUC...",
]

_LOW_ACCURACY_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pipeline_key(state: AppState) -> str:
    """Return the pipeline key ('embedding', 'local', or 'gee') for *state*."""
    if state.model_type == "embedding":
        return "embedding"
    if state.data_mode == "deepdive":
        return "local"
    return "gee"


def _get_steps(pipeline_key: str) -> list[str]:
    """Return the progress step labels for *pipeline_key*."""
    if pipeline_key == "embedding":
        return _EMBEDDING_STEPS
    if pipeline_key == "local":
        return _LOCAL_STEPS
    return _GEE_STEPS


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(state: AppState, on_next: Callable, on_back: Callable) -> None:
    """Render Step 4: Run SDM.

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
    _run_btn_ref: list[ui.button] = []
    _next_btn_ref: list[ui.button] = []
    _status_label_ref: list[ui.label] = []
    _progress_bar_ref: list[ui.linear_progress] = []
    _metrics_section_ref: list[ui.element] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_next_button() -> None:
        """Enable Next button once a classified image is available."""
        if _next_btn_ref:
            _next_btn_ref[0].set_enabled(state.classified_img is not None)

    def _show_metrics() -> None:
        """Populate the metrics section from state.results_df."""
        if not _metrics_section_ref or state.results_df is None:
            return

        container = _metrics_section_ref[0]
        container.clear()

        with container:
            if "roc_auc" in state.results_df.columns:
                mean_auc = state.results_df["roc_auc"].mean()
                row_text = f"Mean ROC-AUC: {mean_auc:.3f}"
                ui.label(row_text).classes("font-semibold")
                if mean_auc < _LOW_ACCURACY_THRESHOLD:
                    ui.badge("Low accuracy", color="warning").classes("text-xs")

            if "overall_accuracy" in state.results_df.columns:
                acc = state.results_df["overall_accuracy"].mean()
                row_text = f"Overall Accuracy: {acc:.3f}"
                ui.label(row_text).classes("font-semibold")
                if acc < _LOW_ACCURACY_THRESHOLD:
                    ui.badge("Low accuracy", color="warning").classes("text-xs")

        container.set_visibility(True)

    def _run_sdm() -> None:
        """Kick off the SDM pipeline in a thread executor."""

        if _run_btn_ref:
            _run_btn_ref[0].set_enabled(False)

        if _metrics_section_ref:
            _metrics_section_ref[0].set_visibility(False)

        pipeline_key = _get_pipeline_key(state)
        steps = _get_steps(pipeline_key)
        n_steps = len(steps)

        async def _run() -> None:
            loop = asyncio.get_event_loop()

            # Show progress bar
            if _progress_bar_ref:
                _progress_bar_ref[0].set_visibility(True)
                _progress_bar_ref[0].value = 0.0

            # Progress simulation coroutine that runs while computation proceeds.
            async def _simulate_progress() -> None:
                for idx, label in enumerate(steps):
                    if _status_label_ref:
                        _status_label_ref[0].set_text(label)
                        _status_label_ref[0].set_visibility(True)
                    if _progress_bar_ref:
                        _progress_bar_ref[0].value = idx / n_steps
                    await asyncio.sleep(1.5)

            # Select the appropriate service call
            if pipeline_key == "embedding":
                service_fn = lambda: sdm_service.run_embedding(state)  # noqa: E731
            elif pipeline_key == "local":
                service_fn = lambda: sdm_service.run_local(state)  # noqa: E731
            else:
                service_fn = lambda: sdm_service.run_gee(state)  # noqa: E731

            # Run computation and progress simulation concurrently
            compute_task = loop.run_in_executor(None, service_fn)
            progress_task = asyncio.ensure_future(_simulate_progress())

            try:
                await compute_task
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                progress_task.cancel()
                if _status_label_ref:
                    _status_label_ref[0].set_text(f"Error: {exc}")
                    _status_label_ref[0].set_visibility(True)
                if _progress_bar_ref:
                    _progress_bar_ref[0].set_visibility(False)
                return
            finally:
                if _run_btn_ref:
                    _run_btn_ref[0].set_enabled(True)

            # Computation succeeded — finish progress display
            progress_task.cancel()
            if _status_label_ref:
                _status_label_ref[0].set_text("Done.")
            if _progress_bar_ref:
                _progress_bar_ref[0].value = 1.0

            _show_metrics()
            _refresh_next_button()

        asyncio.ensure_future(_run())

    # ------------------------------------------------------------------
    # Pipeline info
    # ------------------------------------------------------------------

    pipeline_key = _get_pipeline_key(state)
    pipeline_label = _PIPELINE_LABELS[pipeline_key]

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-xl mx-auto"):
        ui.label("Step 4 — Run SDM").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-4"):
            # 1. Pipeline info label
            ui.label(f"Pipeline: {pipeline_label}").classes(
                "text-sm text-gray-600 italic"
            )

            # 2. Run SDM button
            run_btn = ui.button("Run SDM", on_click=_run_sdm).classes("w-full")
            _run_btn_ref.append(run_btn)

            # 3. Progress: status label + linear progress bar
            status_label = ui.label("").classes("text-sm")
            status_label.set_visibility(False)
            _status_label_ref.append(status_label)

            progress_bar = ui.linear_progress(value=0.0).classes("w-full")
            progress_bar.set_visibility(False)
            _progress_bar_ref.append(progress_bar)

            # 4. Metrics section (hidden until run completes successfully)
            with ui.column().classes(
                "w-full gap-2 pl-4 border-l-2 border-green-400"
            ) as metrics_section:
                ui.label("Results").classes("font-semibold text-sm text-gray-600")
            metrics_section.set_visibility(False)
            _metrics_section_ref.append(metrics_section)

            # Pre-populate metrics if results already exist (re-render case)
            if state.results_df is not None:
                _show_metrics()

            # 5. Navigation buttons
            with ui.row().classes("w-full justify-between mt-2"):
                ui.button("← Back", on_click=on_back)

                next_btn = ui.button("Next →", on_click=on_next)
                _next_btn_ref.append(next_btn)

            # Reflect initial state
            _refresh_next_button()

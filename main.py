"""SDM Toolbox — NiceGUI entry point.

Initialises Google Earth Engine on first page load, then renders a 5-step
stepper that guides the user through the species-distribution modelling
workflow.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from pathlib import Path

# Set multiprocessing start method before any imports that use it
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass  # Already set

from nicegui import app as nicegui_app, ui, context as nicegui_context  # noqa: F401

_CONFIG_PATH = Path.home() / ".sdm-toolbox" / "config.json"


def _load_last_project() -> str:
    try:
        return json.loads(_CONFIG_PATH.read_text()).get("gee_project", "")
    except Exception:
        return ""


def _load_gbif_credentials() -> tuple[str, str]:
    """Load saved GBIF credentials from config."""
    try:
        data = json.loads(_CONFIG_PATH.read_text())
        return data.get("gbif_user", ""), data.get("gbif_pwd", "")
    except Exception:
        return "", ""


def _save_last_project(project_id: str) -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        try:
            data = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
        data["gee_project"] = project_id
        _CONFIG_PATH.write_text(json.dumps(data))
    except Exception:
        pass


def _save_gbif_credentials(user: str, pwd: str) -> None:
    """Save GBIF credentials to config."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        try:
            data = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
        data["gbif_user"] = user
        data["gbif_pwd"] = pwd
        _CONFIG_PATH.write_text(json.dumps(data))
    except Exception:
        pass


from app.services import gee_service
from app.state import AppState
from app.steps import step1_region, step2_data, step3_model, step4_run, step5_results


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, object]] = [
    ("Region & Species", step1_region.render),
    ("Data Source", step2_data.render),
    ("Layers & Model", step3_model.render),
    ("Run SDM", step4_run.render),
    ("Results", step5_results.render),
]

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@ui.page("/")
async def main_page() -> None:
    """Main page: render skeleton immediately, init GEE in background."""

    state = AppState()
    current_step = [0]
    content_ref: list[ui.column] = []
    step_pill_refs: list[list[ui.label]] = [[]]

    # ------------------------------------------------------------------
    # Layout skeleton — rendered immediately so the page loads fast
    # ------------------------------------------------------------------

    with ui.column().classes("w-full min-h-screen p-4"):
        stepper_header = ui.row().classes("w-full justify-center gap-2 mb-6 flex-wrap")
        content = ui.column().classes("w-full")
        content_ref.append(content)

    # ------------------------------------------------------------------
    # GEE init (background task — runs after page is rendered)
    # ------------------------------------------------------------------

    _client = nicegui_context.client

    async def _init_gee_background() -> None:
        loop = asyncio.get_event_loop()
        with _client:
            # Spinner while connecting
            with ui.dialog() as spinner_dlg:
                with ui.card().classes("p-4"):
                    with ui.row().classes("items-center gap-3"):
                        ui.spinner(size="md")
                        ui.label("Initialising Google Earth Engine…").classes(
                            "text-base"
                        )
            spinner_dlg.open()

            first_error = ""
            try:
                last_project = _load_last_project() or None
                await loop.run_in_executor(
                    None,
                    lambda: gee_service.initialize_gee(project=last_project),
                )
                spinner_dlg.close()
                render_step(0)
                return
            except RuntimeError as exc:
                first_error = str(exc)

            spinner_dlg.close()

            # Ask for project ID
            project_id: list[str] = [""]
            ready: asyncio.Event = asyncio.Event()

            with ui.dialog() as proj_dlg:
                with ui.card().classes("p-6 w-96"):
                    ui.label("GEE Project Required").classes("text-lg font-bold mb-2")
                    ui.label(first_error).classes("text-sm text-gray-500 mb-3")
                    ui.label(
                        "Enter your Google Cloud project ID registered with Earth Engine:"
                    ).classes("text-sm mb-2")
                    inp = ui.input(
                        placeholder="e.g. my-gee-project-123",
                        value=_load_last_project(),
                    ).classes("w-full mb-4")

                    def _submit():
                        project_id[0] = inp.value.strip()
                        _save_last_project(project_id[0])
                        proj_dlg.close()
                        ready.set()

                    ui.button("Connect", on_click=_submit).classes("w-full")

            proj_dlg.open()
            await ready.wait()

            if not project_id[0]:
                return

            with ui.dialog() as spinner_dlg2:
                with ui.card().classes("p-4"):
                    with ui.row().classes("items-center gap-3"):
                        ui.spinner(size="md")
                        ui.label("Connecting…").classes("text-base")
            spinner_dlg2.open()

            try:
                await loop.run_in_executor(
                    None, lambda: gee_service.initialize_gee(project=project_id[0])
                )
                spinner_dlg2.close()
                render_step(0)
            except RuntimeError as exc:
                spinner_dlg2.close()
                ui.notify(str(exc), type="negative", timeout=0, close_button=True)
                with content_ref[0]:
                    ui.label("Google Earth Engine could not be initialised.").classes(
                        "text-red-600 font-semibold text-lg mt-8"
                    )
                    ui.label(str(exc)).classes("text-sm text-gray-600 mt-2")

    # ------------------------------------------------------------------
    # Stepper-header helpers
    # ------------------------------------------------------------------

    def _build_stepper_header(active_index: int) -> None:
        """(Re)build pill labels inside *stepper_header* for *active_index*."""
        stepper_header.clear()
        step_pill_refs[0].clear()
        with stepper_header:
            for i, (name, _) in enumerate(STEPS):
                label_text = f"{i + 1}. {name}"
                if i == active_index:
                    pill = ui.label(label_text).classes(
                        "px-3 py-1 rounded-full text-sm font-semibold "
                        "bg-blue-600 text-white"
                    )
                else:
                    pill = ui.label(label_text).classes(
                        "px-3 py-1 rounded-full text-sm bg-gray-200 text-gray-500"
                    )
                step_pill_refs[0].append(pill)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def render_step(step_index: int) -> None:
        """Clear the content area and render *step_index*."""
        content_ref[0].clear()
        _build_stepper_header(step_index)

        with content_ref[0]:
            _, fn = STEPS[step_index]

            if step_index == 0:
                fn(state, on_next=lambda: navigate(1))
            elif step_index == 4:
                fn(state, on_back=lambda: navigate(3))
            else:
                fn(
                    state,
                    on_next=lambda si=step_index: navigate(si + 1),
                    on_back=lambda si=step_index: navigate(si - 1),
                )

    def navigate(new_index: int) -> None:
        """Move to *new_index* and re-render."""
        current_step[0] = new_index
        render_step(new_index)

    # ------------------------------------------------------------------
    # Kick off GEE init in the background (page already rendered above)
    # ------------------------------------------------------------------

    asyncio.ensure_future(_init_gee_background())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui.run(
        native=True,
        title="SDM Toolbox",
        window_size=(1200, 800),
        reload=False,
    )

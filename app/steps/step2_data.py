"""Step 2 — Data Source selection and species occurrence fetch."""

from __future__ import annotations

import asyncio
from typing import Callable, TYPE_CHECKING

import folium
from nicegui import ui

from app.map_server import make_iframe, set_iframe_map
from app.state import AppState
from app.services import gbif_service

if TYPE_CHECKING:
    import main as app_main

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

_MODES: dict[str, str] = {
    "explore": "Explore (Fast, ~300 points)",
    "deepdive": "Deep Dive (All records, slower)",
    "own": "Own Dataset",
}

_AUSTRIA_ANIMALIA_KEY = "0013960-260226173443078"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(state: AppState, on_next: Callable, on_back: Callable) -> None:
    """Render Step 2: Data Source.

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
    _own_section_ref: list[ui.element] = []
    _dataset_key_input_ref: list[ui.input] = []
    _gbif_user_input_ref: list[ui.input] = []
    _gbif_pwd_input_ref: list[ui.input] = []
    _save_creds_checkbox_ref: list[ui.checkbox] = []
    _status_label_ref: list[ui.label] = []
    _map_iframe_ref: list[ui.element] = []
    _fetch_btn_ref: list[ui.button] = []
    _progress_ref: list[ui.spinner] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_next_button() -> None:
        """Enable/disable Next button based on whether species_gdf is populated."""
        has_data = state.species_gdf is not None and len(state.species_gdf) > 0
        if _next_btn_ref:
            _next_btn_ref[0].set_enabled(has_data)

    def _refresh_own_section() -> None:
        """Show/hide the Own Dataset section based on current mode."""
        if _own_section_ref:
            _own_section_ref[0].set_visibility(state.data_mode == "own")

    def _on_mode_change(e) -> None:
        state.data_mode = e.value
        _refresh_own_section()

    def _on_dataset_key_change(e) -> None:
        state.dataset_key = e.value or ""

    def _on_gbif_user_change(e) -> None:
        state.gbif_user = e.value or ""

    def _on_gbif_pwd_change(e) -> None:
        state.gbif_pwd = e.value or ""

    def _on_save_creds_change(e) -> None:
        if e.value:
            import main as app_main

            app_main._save_gbif_credentials(state.gbif_user, state.gbif_pwd)

    def _set_austria_key() -> None:
        state.dataset_key = _AUSTRIA_ANIMALIA_KEY
        if _dataset_key_input_ref:
            _dataset_key_input_ref[0].value = _AUSTRIA_ANIMALIA_KEY

    def _build_folium_html(gdf) -> str:
        """Build a standalone folium map HTML string centred on gdf bounds."""
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        for _, row in gdf.iterrows():
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

    def _update_map(gdf) -> None:
        """Push new folium HTML into the persistent iframe element."""
        if _map_iframe_ref:
            set_iframe_map(_map_iframe_ref[0], _build_folium_html(gdf))

    def _fetch_data() -> None:
        """Fetch presence data synchronously in a thread executor."""

        if _status_label_ref:
            _status_label_ref[0].set_text("Fetching data…")
            _status_label_ref[0].set_visibility(True)

        if _progress_ref:
            _progress_ref[0].set_visibility(True)

        if _fetch_btn_ref:
            _fetch_btn_ref[0].set_enabled(False)

        async def _run() -> None:
            loop = asyncio.get_event_loop()
            try:

                def _fetch():
                    return gbif_service.fetch_presences(
                        state.data_mode,
                        state.species,
                        state.country_code,
                        state.year_start,
                        state.year_end,
                        state.dataset_key,
                        state.gbif_user,
                        state.gbif_pwd,
                    )

                gdf = await loop.run_in_executor(
                    None,
                    lambda: _fetch(),
                )
                state.species_gdf = gdf
                n = len(gdf)
                if _status_label_ref:
                    _status_label_ref[0].set_text(
                        f"Found {n} presence points"
                        if n > 0
                        else "No presence points found."
                    )
                if n > 0:
                    _update_map(gdf)
            except Exception as exc:  # noqa: BLE001
                state.species_gdf = None
                if _status_label_ref:
                    _status_label_ref[0].set_text(f"Error: {exc}")
            finally:
                if _fetch_btn_ref:
                    _fetch_btn_ref[0].set_enabled(True)
                if _progress_ref:
                    _progress_ref[0].set_visibility(False)
                _refresh_next_button()

        asyncio.ensure_future(_run())

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-xl mx-auto"):
        ui.label("Step 2 — Data Source").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-4"):
            # 1. Mode radio group
            ui.radio(
                options=_MODES,
                value=state.data_mode,
                on_change=_on_mode_change,
            ).classes("w-full")

            # 2. Own Dataset section (conditionally visible)
            with ui.column().classes(
                "w-full gap-2 pl-4 border-l-2 border-gray-300"
            ) as own_section:
                ui.label("Own Dataset").classes("font-semibold text-sm text-gray-600")

                dataset_key_input = ui.input(
                    label="GBIF Dataset Key",
                    value=state.dataset_key,
                    on_change=_on_dataset_key_change,
                ).classes("w-full")
                _dataset_key_input_ref.append(dataset_key_input)

                ui.button(
                    "🇦🇹 Austria Animalia",
                    on_click=_set_austria_key,
                ).classes("self-start")

                ui.label("GBIF Credentials").classes(
                    "font-semibold text-sm text-gray-600 mt-2"
                )

                import main as app_main

                saved_user, saved_pwd = app_main._load_gbif_credentials()
                state.gbif_user = state.gbif_user or saved_user
                state.gbif_pwd = state.gbif_pwd or saved_pwd

                gbif_user_input = ui.input(
                    label="GBIF Username",
                    value=state.gbif_user,
                    on_change=_on_gbif_user_change,
                ).classes("w-full")
                _gbif_user_input_ref.append(gbif_user_input)

                gbif_pwd_input = ui.input(
                    label="GBIF Password",
                    value=state.gbif_pwd,
                    on_change=_on_gbif_pwd_change,
                    password=True,
                    password_toggle_button=True,
                ).classes("w-full")
                _gbif_pwd_input_ref.append(gbif_pwd_input)

                save_creds_checkbox = ui.checkbox(
                    "Save credentials",
                    value=False,
                    on_change=_on_save_creds_change,
                ).classes("self-start")
                _save_creds_checkbox_ref.append(save_creds_checkbox)

            _own_section_ref.append(own_section)
            _refresh_own_section()

            # 3. Fetch button
            fetch_btn = ui.button(
                "Fetch Data",
                on_click=_fetch_data,
            ).classes("w-full")
            _fetch_btn_ref.append(fetch_btn)

            # 4. Progress spinner (hidden until fetching)
            with ui.row().classes("w-full justify-center"):
                progress_spinner = ui.spinner(size="lg")
                progress_spinner.set_visibility(False)
                _progress_ref.append(progress_spinner)

            # 5. Status label (hidden until first fetch)
            status_label = ui.label("").classes("text-sm")
            status_label.set_visibility(False)
            _status_label_ref.append(status_label)

            # 6. Map preview (persistent iframe element updated via set_iframe_map)
            map_iframe = make_iframe(height="420px")
            _map_iframe_ref.append(map_iframe)
            if state.species_gdf is not None and len(state.species_gdf) > 0:
                status_label.set_text(f"Found {len(state.species_gdf)} presence points")
                status_label.set_visibility(True)
                set_iframe_map(map_iframe, _build_folium_html(state.species_gdf))

            # 6. Navigation buttons
            with ui.row().classes("w-full justify-between mt-2"):
                ui.button("← Back", on_click=on_back)

                next_btn = ui.button(
                    "Next →",
                    on_click=on_next,
                )
                _next_btn_ref.append(next_btn)

            # Reflect initial state
            _refresh_next_button()

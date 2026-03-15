"""Step 1 — Region & Species selection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import geopandas as gpd
import httpx
from nicegui import ui

from app.state import AppState

# ---------------------------------------------------------------------------
# Module-level constants & data
# ---------------------------------------------------------------------------

COUNTRY_OPTIONS: list[str] = [
    "AT",
    "DE",
    "CH",
    "FR",
    "IT",
    "ES",
    "PL",
    "CZ",
    "HU",
    "RO",
    "SE",
    "NO",
    "FI",
    "DK",
    "NL",
    "BE",
    "PT",
    "GR",
    "HR",
    "SK",
    "SI",
    "BG",
    "LT",
    "LV",
    "EE",
]

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_NUTS2: gpd.GeoDataFrame = gpd.read_file(
    _ASSETS / "NUTS_RG_01M_2024_4326_LEVL_2.geojson"
)


def _nuts2_options_for_country(country_code: str) -> dict[str, str]:
    """Return {NUTS_NAME: NUTS_NAME} for *country_code*.

    Uses NUTS_NAME as both key and label so that state.county_name stores the
    name that get_aoi_from_nuts() expects (it filters by NUTS_2.NUTS_NAME).
    """
    subset = _NUTS2[_NUTS2["CNTR_CODE"] == country_code]
    return {row["NUTS_NAME"]: row["NUTS_NAME"] for _, row in subset.iterrows()}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(state: AppState, on_next: Callable) -> None:
    """Render Step 1: Region & Species.

    Parameters
    ----------
    state:
        Shared ``AppState`` instance; mutations are reflected app-wide.
    on_next:
        Callable invoked when the user clicks the "Next →" button.
    """

    # Mutable container used as a closure reference so inner functions can
    # reach widgets that are created after the function is defined.
    _debounce_tasks: list[asyncio.Task] = []
    _next_btn_ref: list[ui.button] = []
    _county_select_ref: list[ui.select] = []

    # ------------------------------------------------------------------
    # Helpers defined before widget creation so ordering is unambiguous.
    # ------------------------------------------------------------------

    def _refresh_next_button() -> None:
        """Enable/disable the Next button based on required fields."""
        enabled = bool(state.species) and bool(state.country_code)
        if _next_btn_ref:
            _next_btn_ref[0].set_enabled(enabled)

    async def _fetch_suggestions(value: str) -> None:
        """Call GBIF suggest API and populate the suggestions dropdown."""
        if not value or len(value) < 2:
            if _suggestions_ref:
                _suggestions_ref[0].set_options([])
                _suggestions_ref[0].set_visibility(False)
            return
        url = f"https://api.gbif.org/v1/species/suggest?q={value}&limit=5"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            names = [
                item.get("canonicalName") or item.get("scientificName", "")
                for item in data
            ]
            names = [n for n in names if n]
        except Exception:
            names = []
        if _suggestions_ref:
            _suggestions_ref[0].set_options(names)
            _suggestions_ref[0].set_visibility(bool(names))

    def _on_species_change(e) -> None:
        """Handle every keystroke in the species input."""
        state.species = e.value or ""
        _refresh_next_button()
        # Cancel any pending debounce task and schedule a new one.
        for t in _debounce_tasks:
            t.cancel()
        _debounce_tasks.clear()

        async def _debounced() -> None:
            await asyncio.sleep(0.5)
            await _fetch_suggestions(state.species)

        _debounce_tasks.append(asyncio.ensure_future(_debounced()))

    def _on_suggestion_select(e) -> None:
        """Fill the species input when user picks a GBIF suggestion."""
        if e.value and _species_input_ref:
            state.species = e.value
            _species_input_ref[0].value = e.value
            _suggestions_ref[0].set_visibility(False)
            _refresh_next_button()

    def _on_country_change(e) -> None:
        """Repopulate the county dropdown and update state."""
        state.country_code = e.value or ""
        state.county_name = ""
        if _county_select_ref:
            new_opts = _nuts2_options_for_country(state.country_code)
            _county_select_ref[0].set_options(new_opts)
            _county_select_ref[0].value = None
        _refresh_next_button()

    def _on_county_change(e) -> None:
        state.county_name = e.value or ""

    def _on_year_change(e) -> None:
        try:
            state.year_start = int(e.value)
        except (TypeError, ValueError):
            pass

    def _on_year_end_change(e) -> None:
        try:
            state.year_end = int(e.value)
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Widget references (populated during construction below).
    # ------------------------------------------------------------------
    _species_input_ref: list[ui.input] = []
    _suggestions_ref: list[ui.select] = []

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    with ui.card().classes("w-full max-w-xl mx-auto"):
        ui.label("Step 1 — Region & Species").classes("text-xl font-bold mb-4")

        with ui.column().classes("w-full gap-4"):
            # 1. Species input
            species_input = ui.input(
                label="Species name",
                value=state.species,
                on_change=_on_species_change,
            ).classes("w-full")
            _species_input_ref.append(species_input)

            # GBIF autocomplete suggestions (hidden until results arrive)
            suggestions_select = ui.select(
                label="Suggestions (select to fill)",
                options=[],
                value=None,
                on_change=_on_suggestion_select,
            ).classes("w-full")
            suggestions_select.set_visibility(False)
            _suggestions_ref.append(suggestions_select)

            # 2. Country dropdown
            country_select = ui.select(
                label="Country",
                options=COUNTRY_OPTIONS,
                value=state.country_code or None,
                on_change=_on_country_change,
            ).classes("w-full")

            # 3. County (NUTS-2) dropdown — pre-populated if country already set
            initial_county_opts = (
                _nuts2_options_for_country(state.country_code)
                if state.country_code
                else {}
            )
            county_select = ui.select(
                label="County (NUTS-2)",
                options=initial_county_opts,
                value=state.county_name or None,
                on_change=_on_county_change,
            ).classes("w-full")
            _county_select_ref.append(county_select)

            # 4. Year range selector
            with ui.row().classes("w-full gap-4"):
                ui.number(
                    label="Year from",
                    value=state.year_start,
                    min=2000,
                    max=2025,
                    step=1,
                    on_change=_on_year_change,
                ).classes("w-full")
                ui.number(
                    label="Year to",
                    value=state.year_end,
                    min=2000,
                    max=2025,
                    step=1,
                    on_change=_on_year_end_change,
                ).classes("w-full")

            # 5. Next button — disabled until required fields are filled
            next_btn = ui.button(
                "Next →",
                on_click=on_next,
            ).classes("mt-2 self-end")
            _next_btn_ref.append(next_btn)

            # Reflect initial state (state may be pre-populated on re-render)
            _refresh_next_button()

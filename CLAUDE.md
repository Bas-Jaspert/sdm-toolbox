# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDM-Toolbox is a Python package for interactive species distribution modeling (SDM) using Google Earth Engine (GEE), GBIF occurrence data, and scikit-learn. It has **two interfaces**:

1. **NiceGUI desktop app** (`main.py`) — the primary interface, a guided 5-step stepper
2. **Jupyter notebook** (`sdm_workflow.ipynb`) — advanced users who want full control


## Running the App

```bash
# NiceGUI desktop app (primary)
uv run python main.py

# Notebook (advanced)
jupyter notebook sdm_workflow.ipynb
```

On first run, the app opens a GEE project ID dialog. The last-used project ID is saved to `~/.sdm-toolbox/config.json` and pre-filled on subsequent runs.

## Architecture

```
toolbox/
├── main.py                        # NiceGUI entry point: ui.run(native=True)
├── app/
│   ├── state.py                   # AppState dataclass (single source of truth)
│   ├── map_server.py              # Folium HTML → base64 data-URL iframe helper
│   ├── steps/
│   │   ├── step1_region.py        # Species (GBIF autocomplete), country, NUTS-2 county, year
│   │   ├── step2_data.py          # Data source selection + GBIF fetch + map preview
│   │   ├── step3_model.py         # GEE layer selection + model hyperparameters
│   │   ├── step4_run.py           # SDM execution + progress + metrics
│   │   └── step5_results.py       # Classification map, feature importance, what-if, export
│   └── services/
│       ├── gee_service.py         # GEE OAuth init + layer loader wrapper
│       ├── gbif_service.py        # Explore / Deep Dive / Own Dataset fetch modes
│       └── sdm_service.py         # Three SDM pipelines (GEE-native, sklearn, embedding)
├── toolbox/
│   ├── utils.py                   # UNCHANGED — shared core logic
│   ├── stat_functions.py          # UNCHANGED
│   └── __init__.py
├── tests/
│   ├── conftest.py
│   ├── test_services/
│   │   └── test_state.py
│   └── test_utils/
│       ├── test_aoi.py
│       ├── test_background.py
│       └── test_compute_sdm.py
├── assets/                        # NUTS GeoJSONs, background_data.csv
└── sdm_workflow.ipynb             # UNCHANGED — advanced user interface
```

**`toolbox/utils.py` must never be modified** — it is shared with the notebook workflow.

### 5-Step Workflow

| Step | File | Purpose |
|------|------|---------|
| 1 | `step1_region.py` | Species name (GBIF autocomplete), country (ISO-2), NUTS-2 county (AOI only), year |
| 2 | `step2_data.py` | Mode: Explore (≤300 records, no year filter) / Deep Dive (paginated) / Own Dataset (Parquet cache) |
| 3 | `step3_model.py` | Initialize GEE layers, select predictors, choose model + hyperparameters |
| 4 | `step4_run.py` | Run SDM pipeline, show accuracy metric |
| 5 | `step5_results.py` | Classification map, feature importances (RF), what-if panel, Google Drive export |

### State Management

`app/state.py` — single `AppState` dataclass passed through all steps. Each step validates required fields before enabling the Next button.

### SDM Pipelines (`sdm_service.py`)

Three pipelines selected automatically based on `state.data_mode`:

- **`run_gee()`** — Explore / Own Dataset: server-side GEE RF or Maxent. Trains in CLASSIFICATION mode for `errorMatrix` accuracy, then switches to PROBABILITY mode for the suitability map. Feature importances via `classifier.explain()["importance"]`.
- **`run_local()`** — Deep Dive: sklearn RF/Maxent trained locally on downloaded features, then re-applied via GEE for classification.
- **`run_embedding()`** — Embedding: cosine similarity dot-product against Google Satellite Embedding V1.

Background pseudo-absences: equal-sized random sample from `assets/background_data.csv`. County selection affects **classification AOI only** — GBIF data is always fetched country-wide.

### Map Display

Folium maps are rendered as `ui.element('iframe')` with the full map HTML embedded as a `data:text/html;base64,...` src. This is necessary because:
- `ui.html()` blocks `<script>` tags (Leaflet.js uses them)
- pywebview doesn't reliably render iframes injected via `v-html`

Never use `ui.html()` for folium maps. Use `make_iframe()` + `set_iframe_map()` from `app/map_server.py`.

Classification maps use the RdYlBu-reversed palette (blue = low suitability → red = high).

### Key Patterns

- **Async blocking calls**: wrap in `loop.run_in_executor(None, fn)` inside `async def _run()` scheduled with `asyncio.ensure_future()`
- **NiceGUI event handlers**: always use `on_change=handler` in widget constructors, never `.on("update:model-value", handler)` — the latter gives `GenericEventArguments` without `.value`
- **Client context in background tasks**: capture `_client = nicegui_context.client` in the page handler and use `with _client:` inside background tasks spawned outside button click handlers
- **Widget refs**: use `list[T]` containers (e.g. `_btn_ref: list[ui.button] = []`) for closure access to widgets

## Dependencies

Managed via `uv` / `pyproject.toml`. Key packages:

- `earthengine-api>=0.1.390`, `geemap>=0.32.0`, `folium>=0.18`
- `scikit-learn>=1.8`, `scipy>=1.17`, `geopandas>=0.14`, `shapely>=2.1`
- `nicegui>=2.0`, `pywebview>=5.0`, `pyqt6>=6.10`, `pyqt6-webengine>=6.10`, `qtpy>=2.4`
- `pygbif>=0.6.6`, `httpx>=0.27`

## Test Suite

Run the automated test suite with:

```bash
uv run pytest
```

Test files:
- `tests/test_services/test_state.py` — AppState dataclass validation
- `tests/test_utils/test_aoi.py` — AOI / NUTS-2 helpers
- `tests/test_utils/test_background.py` — pseudo-absence sampling
- `tests/test_utils/test_compute_sdm.py` — SDM pipeline unit tests

Manual integration test: `uv run python main.py` with *Lagopus muta*, country AT.

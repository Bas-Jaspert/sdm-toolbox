# Design: Configurable Analysis Resolution

**Date:** 2026-03-22
**Status:** Approved

## Context

The SDM pipelines hardcode `scale=30` metres for GEE feature sampling and spatial prediction. Users working with coarser predictor datasets (e.g. ERA5, BioClim) or needing faster computation want to reduce resolution; users working primarily with Sentinel-2 may want 10 m. A single dropdown in the Step 3 hyperparameters panel exposes this control.

## Constraint

`toolbox/utils.py` is frozen (shared with the notebook workflow). The `classify_image_aoi()` function it contains hardcodes `scale=30` for prediction in the Deep Dive (local sklearn) pipeline.

For `run_local()`, the flow is:
1. Each layer in `state.layer_stack` is pre-reprojected at `state.resolution` before being passed to `get_species_features()`, so `sampleRegions()` inside that function inherits the user-selected scale.
2. `get_species_features()` returns the pre-reprojected `predictors` image (at user scale).
3. `classify_image_aoi()` (frozen) then calls `image.reproject(crs="EPSG:4326", scale=30)` on that image, overriding the user scale back to 30 m for prediction. This is intentional — the frozen function cannot be changed.

| Pipeline | Sampling | Prediction |
|---|---|---|
| GEE (Explore / Own / Upload) | user scale | user scale |
| Local (Deep Dive) | user scale | 30 m (overridden by frozen `classify_image_aoi`) |
| Embedding | 30 m (fixed) | native embedding res |

## State

Add one field to `app/state.py`:

```python
resolution: int = 30  # metres; GEE sampling and prediction scale
```

Allowed values: 10, 30, 100, 300, 1000. Enforced by the UI dropdown.

## UI

A `ui.select` added inside the existing hyperparameters panel in `app/steps/step3_model.py`, after Train Size:

- **Label:** `Resolution (m)`
- **Options:** `{10: "10 m", 30: "30 m", 100: "100 m", 300: "300 m", 1000: "1 000 m"}`
  Keys are `int`; NiceGUI returns the key directly so no cast is needed.
- **Default:** `state.resolution` (30 on first run)
- **Handler:**
  ```python
  _RESOLUTION_OPTIONS = {10, 30, 100, 300, 1000}

  def _on_resolution_change(e) -> None:
      if e.value in _RESOLUTION_OPTIONS:
          state.resolution = e.value
  ```
- Sits inside `hyperparam_section` — automatically hidden when embedding mode is selected via the existing `_refresh_hyperparam_section()` logic.

## Pipeline Changes

All changes in `app/services/sdm_service.py`. Replace hardcoded `scale=30` with `state.resolution`:

| Function | Location | Change |
|---|---|---|
| `run_gee()` | presence `sampleRegions` | `scale=30` → `scale=state.resolution` |
| `run_gee()` | background `sampleRegions` | `scale=30` → `scale=state.resolution` |
| `run_gee()` | prediction reproject | `scale=30` → `scale=state.resolution` |
| `run_local()` | pre-reproject loop | `scale=30` → `scale=state.resolution` |

`run_embedding()` — all `sampleRegions` calls stay at `scale=30`; the embedding mosaic has a fixed native resolution.

## Tests

All tests are in `tests/test_services/test_sdm_service.py`.

### Existing tests — update required

`_make_state()` must gain `state.resolution = 30` so existing assertions remain valid.
The two existing tests (`test_run_gee_reprojects_predictors_before_classify`, `test_run_local_passes_reprojected_layer_stack`) assert `scale=30` — these assertions remain correct once `state.resolution` defaults to 30.

### New tests

1. `test_run_gee_uses_state_resolution` — `state.resolution = 100`; verify both `sampleRegions` calls and the `reproject` call each use `scale=100`
2. `test_run_local_uses_state_resolution` — `state.resolution = 100`; verify the pre-reproject loop uses `scale=100`
3. `test_run_embedding_ignores_resolution` — `state.resolution = 100`; verify **all** `sampleRegions` calls in `run_embedding()` still use `scale=30`

All tests follow the existing mock pattern in `test_sdm_service.py`.

## Files Touched

| File | Change |
|---|---|
| `app/state.py` | Add `resolution: int = 30` |
| `app/steps/step3_model.py` | Add `_RESOLUTION_OPTIONS` constant + resolution dropdown in hyperparam panel |
| `app/services/sdm_service.py` | Replace 4× `scale=30` with `state.resolution` |
| `tests/test_services/test_sdm_service.py` | Add `resolution=30` to `_make_state()`; add 3 new tests |

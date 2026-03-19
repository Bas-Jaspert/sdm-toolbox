"""
Google Earth Engine service module.

Provides OAuth-based authentication and layer information retrieval from GEE.
"""

import ee
from toolbox.utils import get_layer_information as utils_get_layer_information


def initialize_gee(project: str | None = None) -> bool:
    """
    Initialize Google Earth Engine with OAuth authentication.

    Tries ee.Initialize(project=project). Authenticates first if no
    credentials are cached yet.

    Parameters
    ----------
    project : str or None
        GEE cloud project ID (e.g. "my-gee-project").  When None, the EE SDK
        uses the project stored in the cached credentials (if any).

    Returns
    -------
    bool
        True if initialization was successful.

    Raises
    ------
    RuntimeError
        If GEE initialization fails after authentication attempt.
    """
    # If already initialized, verify the connection actually works (the project
    # stored in ee state may not be registered for Earth Engine).
    if ee.data.is_initialized():
        try:
            ee.Number(1).getInfo()
            return True
        except Exception:
            pass  # fall through and re-initialize with the supplied project

    try:
        ee.Authenticate()
        ee.Initialize(project=project)
        ee.Number(1).getInfo()  # verify the project is registered for EE
        return True
    except Exception as e:
        raise RuntimeError(str(e)) from e


def get_layer_information(year: int) -> dict:
    """
    Retrieve environmental predictor layers from Google Earth Engine.

    This is a thin wrapper around toolbox.utils.get_layer_information()
    that provides a consistent interface from the gee_service module.

    Parameters
    ----------
    year : int
        The year for which to retrieve layer information.
        Used for time-filtered datasets (ERA5, NDVI, etc.).

    Returns
    -------
    dict
        A dictionary of EE Image objects, keyed by layer name.
        Includes elevation, slope, aspect, vegetation indices,
        climate variables, and more.

    Raises
    ------
    Exception
        If there is an error retrieving or processing layer data from GEE.
    """
    return utils_get_layer_information(year)

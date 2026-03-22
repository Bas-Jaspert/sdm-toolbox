"""Wikipedia REST API service for species summaries."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class SpeciesSummary:
    """Summary data fetched from the Wikipedia REST API.

    Attributes
    ----------
    title : str
        Display title as returned by Wikipedia (may differ from query name).
    extract : str
        Plain-text summary paragraph.
    thumbnail_url : str | None
        URL of the thumbnail image, or ``None`` if unavailable.
    """

    title: str
    extract: str
    thumbnail_url: str | None


async def fetch_species_summary(species_name: str) -> SpeciesSummary:
    """Fetch a Wikipedia summary for the given species name.

    Parameters
    ----------
    species_name : str
        Scientific or common name. Spaces are converted to underscores.

    Returns
    -------
    SpeciesSummary

    Raises
    ------
    httpx.HTTPStatusError
        On 404 (no article) or other non-2xx responses.
    httpx.RequestError
        On network-level failure (timeout, DNS, etc.).
    """
    slug = species_name.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    headers = {"User-Agent": "SDM-Toolbox/1.0 (species distribution modelling research tool)"}
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
    data = resp.json()
    return SpeciesSummary(
        title=data["title"],
        extract=data.get("extract", ""),
        thumbnail_url=data.get("thumbnail", {}).get("source"),
    )

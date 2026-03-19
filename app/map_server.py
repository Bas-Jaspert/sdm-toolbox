"""Map helpers: encode folium HTML and set it on a NiceGUI iframe element.

Using ui.element('iframe') with a data-URL src bypasses both:
- NiceGUI's <script> tag sanitiser (which blocks ui.html with scripts)
- Any iframe rendering issues in pywebview
"""

from __future__ import annotations

import base64

from nicegui import ui


def folium_to_b64(html) -> str:
    """Return base64-encoded folium HTML for use as a data URL src."""
    if not isinstance(html, str):
        raise ValueError(f"Expected string, got {type(html).__name__}")
    return base64.b64encode(html.encode("utf-8")).decode()


def make_iframe(height: str = "420px") -> ui.element:
    """Create an empty iframe element sized for map display."""
    return ui.element("iframe").style(
        f"width:100%;height:{height};border:none;display:block;"
    )


def set_iframe_map(elem: ui.element, html: str) -> None:
    """Encode *html* as a base64 data URL and push it to *elem*."""
    b64 = folium_to_b64(html)
    elem._props["src"] = f"data:text/html;base64,{b64}"
    elem.update()


# Backwards-compat alias used in step5_results
def save_and_iframe(html: str, height: str = "400px") -> str:
    """Legacy helper — returns base64 data URL string (not used for rendering)."""
    b64 = folium_to_b64(html)
    return f"data:text/html;base64,{b64}"

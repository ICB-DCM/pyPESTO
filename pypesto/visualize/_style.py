"""
Visual style for ``pypesto.visualize``.

Default constants, the ``style_kwargs`` registry, and small cross-module
helpers.

Users override any default per call via ``style_kwargs``, validated against
:data:`_DEFAULTS`::

    waterfall(result, style_kwargs={"mle_color": "tab:purple"})
"""

from __future__ import annotations

import warnings

# Colors — semantic roles
# -----------------------
MLE_COLOR = "#d62728"  # tab:red — best cluster + MLE markers
OUTLIER_COLOR = "#b3b3b3"  # mid-grey — singleton / outlier starts

# Colormaps
# ---------
CMAP_DISCRETE = "tab10"  # qualitative: cluster + per-variable colours

# Style registry
# --------------

_DEFAULTS: dict[str, object] = {
    "mle_color": MLE_COLOR,
    "outlier_color": OUTLIER_COLOR,
    "cmap_discrete": CMAP_DISCRETE,
}


def resolve_style(style_kwargs: dict | None = None) -> dict:
    """Return the effective style dict, merging defaults with caller overrides.

    Parameters
    ----------
    style_kwargs:
        User-supplied overrides. Unknown keys emit a ``UserWarning`` so
        typos surface immediately.

    Returns
    -------
    dict
        Merged style dict with all keys from :data:`_DEFAULTS`, with
        caller overrides applied on top.
    """
    style = dict(_DEFAULTS)
    if style_kwargs:
        unknown = set(style_kwargs) - set(_DEFAULTS)
        if unknown:
            warnings.warn(
                f"Unknown style_kwargs keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(_DEFAULTS)}.",
                UserWarning,
                stacklevel=3,
            )
        style.update(style_kwargs)
    return style

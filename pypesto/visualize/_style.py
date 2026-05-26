"""
Shared visualization constants and helpers for ``pypesto.visualize``.

TODO remove too much info after all PRs have been merged.

Grown incrementally across the PR 1.5 follow-on series: each per-viz PR
adds the constants and helpers its consumers need, in the same diff as
those consumers. The final PR in the series adds :func:`apply_style`
(an opt-in rcParams preset).

Style keys are surfaced to users via the ``style_kwargs`` parameter on
every public plotter, validated against :data:`_DEFAULTS`::

    waterfall(result, style_kwargs={"mle_color": "tab:purple"})

How to extend
-------------
- **Add a constant**: ``UPPER_SNAKE`` name with a 1-line comment on its
  semantic role, under the appropriate section header. Add a new
  section header (``# ===`` block) when the purpose is genuinely new.
- **Add a registry key**: lowercase entry in :data:`_DEFAULTS` referencing
  the constant. Unknown keys passed to :func:`resolve_style` raise
  ``UserWarning`` so typos surface immediately.
- **Add a helper**: under an existing section, or a new one. Helpers that
  cross module boundaries belong here. Module-local helpers stay in
  their module.
"""

from __future__ import annotations

import warnings

# Colors — semantic roles
# -----------------------

# matplotlib ``tab:red``; used for both the best-cluster colour and MLE markers.
MLE_COLOR = "#d62728"

# Neutral mid-grey; isolated (singleton) starts and outlier indicators.
OUTLIER_COLOR = "#b3b3b3"

# Colormaps
# ---------

# Qualitative palette; secondary cluster colours and per-variable colours in
# prediction-trajectory plots are sampled from this.
CMAP_DISCRETE = "tab10"

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
        User-supplied overrides. Unknown keys raise a ``UserWarning`` so
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

"""
Utilities
=========

Package-wide utilities.

"""


def get_condition_label(condition_id: str) -> str:
    """Convert a condition ID to a label.

    Labels for conditions are used at different locations (e.g. ensemble
    prediction code, and visualization code). This method ensures that the same
    condition is labeled identically everywhere.

    Parameters
    ----------
    condition_id:
        The condition ID that will be used to generate a label.

    Returns
    -------
    The condition label.
    """
    return f'condition_{condition_id}'

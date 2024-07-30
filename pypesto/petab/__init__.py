"""
PEtab
=====

pyPESTO support for the PEtab data format.
"""

import warnings

from .importer import PetabImporter

# PEtab and amici are optional dependencies


try:
    import petab
except ImportError:
    warnings.warn(
        "PEtab import requires an installation of petab "
        "(https://github.com/PEtab-dev/PEtab). "
        "Install via `pip3 install petab`.",
        stacklevel=1,
    )
try:
    import amici
except ImportError:
    amici = None

"""
PEtab
=====

pyPESTO support for the PEtab data format.
"""
from .importer import PetabImporter
from .pysb_importer import PetabImporterPysb

# PEtab and amici are optional dependencies

import warnings

try:
    import petab
except ImportError:
    warnings.warn("PEtab import requires an installation of petab "
                  "(https://github.com/PEtab-dev/PEtab). "
                  "Install via `pip3 install petab`.")
try:
    import amici
except ImportError:
    warnings.warn("PEtab import requires an installation of amici "
                  "(https://github.com/AMICI-dev/AMICI). "
                  "Install via `pip3 install amici`.")

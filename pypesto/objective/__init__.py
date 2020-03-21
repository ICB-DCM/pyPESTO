"""
Objective
=========

"""

from .objective import Objective
from .amici_objective import AmiciObjective
from .aggregated import AggregatedObjective
from .options import ObjectiveOptions
from .util import res_to_chi2, sres_to_schi2

# PEtab is an optional dependency
try:
    from .petab_import import PetabImporter
except ModuleNotFoundError:
    PetabImporter = None

__all__ = [
    "Objective",
    "ObjectiveOptions",
    "res_to_chi2",
    "sres_to_schi2",
    "AmiciObjective",
    "AggregatedObjective",
    "PetabImporter"
]

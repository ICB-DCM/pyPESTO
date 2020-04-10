"""
PEtab
=====

Code related to giving pyPESTO support for the PEtab data format.
"""

# PEtab is an optional dependency
try:
    from .petab_importer import PetabImporter
except ModuleNotFoundError:
    PetabImporter = None

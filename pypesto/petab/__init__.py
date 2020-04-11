"""
PEtab
=====

pyPESTO support for the PEtab data format.
"""

# PEtab is an optional dependency
try:
    from .importer import PetabImporter
except ModuleNotFoundError:
    PetabImporter = None

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from .importer import PetabImporter

if TYPE_CHECKING:
    import amici.petab_import_pysb


class PetabImporterPysb(PetabImporter):
    """Import for experimental PySB-based PEtab problems."""

    def __init__(
        self,
        petab_problem: amici.petab_import_pysb.PysbPetabProblem,
        validate_petab: bool = False,
        **kwargs,
    ):
        """
        Initialize importer.

        Parameters
        ----------
        petab_problem:
            Managing access to the model and data.
        validate_petab:
            Flag indicating if the PEtab problem shall be validated.
        kwargs:
            Passed to `PetabImporter.__init__`.
        """
        if "model_name" not in kwargs:
            kwargs["model_name"] = petab_problem.pysb_model.name
        super().__init__(
            petab_problem,
            validate_petab=validate_petab,
            **kwargs,
        )

    def compile_model(self, **kwargs):
        """
        Compile the model.

        If the output folder exists already, it is first deleted.

        Parameters
        ----------
        kwargs: Extra arguments passed to `amici.SbmlImporter.sbml2amici`.

        """
        import amici.petab_import_pysb

        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        amici.petab_import_pysb.import_model_pysb(
            petab_problem=self.petab_problem,
            model_output_dir=self.output_folder,
            **kwargs,
        )

from .importer import PetabImporter
import os
import shutil

try:
    import amici
    import amici.petab_import_pysb
except ImportError:
    pass


class PetabImporterPysb(PetabImporter):
    """Import for experimental PySB-based PEtab problems"""

    def __init__(self,
                 petab_problem: 'amici.petab_import_pysb.PysbPetabProblem',
                 output_folder: str = None):
        """
        petab_problem:
            Managing access to the model and data.
        output_folder:
            Folder to contain the amici model.
        """

        super().__init__(petab_problem,
                         model_name=petab_problem.pysb_model.name,
                         output_folder=output_folder)

    def compile_model(self, **kwargs):
        """
        Compile the model. If the output folder exists already, it is first
        deleted.

        Parameters
        ----------
        kwargs: Extra arguments passed to `amici.SbmlImporter.sbml2amici`.

        """

        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        amici.petab_import_pysb.import_model_pysb(
            petab_problem=self.petab_problem,
            model_output_dir=self.output_folder,
            **kwargs)

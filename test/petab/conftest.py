import os

import benchmark_models_petab as models
import petab
import pytest

import pypesto.petab


@pytest.fixture(scope="session")
def loaded_models():
    """Fixture to load all models used in Tests to be only loaded once."""
    model_names = [
        "Zheng_PNAS2012",
        "Brannmark_JBC2010",
        "Boehm_JProteomeRes2014",
    ]
    objectives = {}
    for model_name in model_names:
        petab_problem = petab.Problem.from_yaml(
            os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
        )
        petab_problem.model_name = model_name
        importer = pypesto.petab.PetabImporter(petab_problem)
        obj = importer.create_objective()
        objectives[model_name] = importer.create_problem(obj)

    return objectives

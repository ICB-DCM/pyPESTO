import pytest
from pypesto.modelselection.row2problem import *
from pypesto.petab import PetabImporter
import petab
import pandas as pd


@pytest.fixture
def modelselection_file_example():
    return "doc/example/model_selection/modelSelectionSpecification_example_" \
           "modelSelection.tsv"


@pytest.fixture
def yaml_file_example():
    return "doc/example/model_selection/example_modelSelection.yaml"


@pytest.mark.xfail  # example does not seem to compile properly yet
def test_row2problem_yaml_string(yaml_file_example):
    petab_problem = petab.Problem.from_yaml(yaml_file_example)

    importer = PetabImporter(petab_problem)
    obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)

    row = pd.Series()
    assert row2problem(row, yaml_file_example) == pypesto_problem


@pytest.mark.xfail  # example does not seem to compile properly yet
def test_row2problem_setting_pars(yaml_file_example,
                                  modelselection_file_example):
    petab_problem = petab.Problem.from_yaml(yaml_file_example)
    table = pd.read_csv(modelselection_file_example, sep="\t", index_col=True)
    row = table.loc["M1"]
    returned_problem = row2problem(row, yaml_file_example)

"""Tests for `pypesto.prediction.AmiciPredictor`."""

import amici
import pypesto
import pypesto.petab
import pypesto.ensemble
import os
import sys
import numpy as np
import pandas as pd
import shutil
import pytest
import libsbml
import petab

from pypesto.predict import (
    AmiciPredictor,
    PredictionConditionResult,
    PredictionResult,
)


@pytest.fixture()
def conversion_reaction_model():
    # read in sbml file
    model_name = "conversion_reaction"
    example_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "doc", "example"
    )
    sbml_file = os.path.join(
        example_dir, model_name, f"model_{model_name}.xml"
    )
    model_output_dir = os.path.join(
        example_dir, "tmp", f"{model_name}_enhanced"
    )

    # try to import the exisiting model, if possible
    try:
        sys.path.insert(0, os.path.abspath(model_output_dir))
        model_module = amici.import_model_module(model_name, model_output_dir)
        model = model_module.getModel()
    except ValueError:
        # read in and adapt the sbml slightly
        if os.path.abspath(model_output_dir) in sys.path:
            sys.path.remove(os.path.abspath(model_output_dir))
        sbml_importer = amici.SbmlImporter(sbml_file)

        # add observables to sbml model
        def create_observable(sbml_model, obs_id):
            # create a parameter, which will get a rule assignmed as observable
            parameter = sbml_model.createParameter()
            parameter.setId(f"observable_{obs_id}")
            parameter.setName(f"observable_{obs_id}")
            parameter.constant = True

            rule = sbml_importer.sbml.createAssignmentRule()
            rule.setId(f"observable_{obs_id}")
            rule.setName(f"observable_{obs_id}")
            rule.setVariable(f"observable_{obs_id}")
            rule.setFormula(obs_id)

        # add initial assignments to sbml model
        def create_intial_assignment(sbml_model, spec_id):
            # create a parameter, which will get a rule assignmed as observable
            parameter = sbml_model.createParameter()
            parameter.setId(f"{spec_id}0")
            parameter.setName(f"{spec_id}0")
            parameter.constant = True

            assignment = sbml_importer.sbml.createInitialAssignment()
            assignment.setSymbol(f"{spec_id}")
            math = (
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>'
                f"{spec_id}0</ci></math>"
            )
            assignment.setMath(libsbml.readMathMLFromString(math))

        for spec in ("A", "B"):
            create_observable(sbml_importer.sbml, spec)
            create_intial_assignment(sbml_importer.sbml, spec)

        # add constant parameters and observables to AMICI model
        constant_parameters = ["A0", "B0"]
        observables = amici.assignmentRules2observables(
            sbml_importer.sbml,  # the libsbml model object
            filter_function=lambda variable: variable.getId().startswith(
                "observable_"
            ),
        )
        # generate the python module for the model.
        sbml_importer.sbml2amici(
            model_name,
            model_output_dir,
            verbose=False,
            observables=observables,
            constant_parameters=constant_parameters,
        )

        # Importing the module and loading the model
        sys.path.insert(0, os.path.abspath(model_output_dir))
        model_module = amici.import_model_module(model_name, model_output_dir)
        model = model_module.getModel()
    except RuntimeError as err:
        print(
            "pyPESTO unit test ran into an error importing the conversion "
            "reaction enhanced model. This may happen due to an old version "
            "of this model being present in your python path (e.g., "
            "incorrect AMICI version comparing to the installed one). "
            "Delete the conversion_reaction_enhanced model from your python "
            "path and retry. Your python path is currently:"
        )
        print(sys.path)
        print("Original error message:")
        raise err

    return model


@pytest.fixture()
def edata_objects(conversion_reaction_model):
    testmodel = conversion_reaction_model

    # set timepoints for which we want to simulate the model
    testmodel.setTimepoints(np.linspace(0, 4, 10))
    testmodel.setParameters(np.array([4.0, 0.4]))
    # Create solver instance
    solver = testmodel.getSolver()

    # create edatas
    rdatas = []
    edatas = []
    fixedParameters = [
        np.array([2.0, 0.0]),
        np.array([0.0, 4.0]),
        np.array([1.0, 1.0]),
    ]
    # create rdatas and edatas from those
    for fp in fixedParameters:
        testmodel.setFixedParameters(amici.DoubleVector(fp))
        rdata = amici.runAmiciSimulation(testmodel, solver)
        rdatas.append(rdata)
        edatas.append(amici.ExpData(rdata, 1.0, 0))

    return testmodel, solver, edatas


def check_outputs(predicted, out, n_cond, n_timepoints, n_obs, n_par):
    # correct output type?
    assert isinstance(predicted, PredictionResult)
    # correct number of predictions?
    assert len(predicted.conditions) == n_cond

    # check whether conversion to dict worked well
    preDict = dict(predicted)
    assert isinstance(preDict, dict)
    assert len(preDict["conditions"]) == n_cond
    for cond in preDict["conditions"]:
        assert isinstance(cond, dict)

    # correct shape for outputs?
    if 0 in out:
        for cond in predicted.conditions:
            assert isinstance(cond, PredictionConditionResult)
            assert isinstance(cond.output, np.ndarray)
            assert cond.output.shape == (n_timepoints, n_obs)
    # correct shape for output sensitivities?
    if 1 in out:
        for cond in predicted.conditions:
            assert isinstance(cond, PredictionConditionResult)
            assert isinstance(cond.output_sensi, np.ndarray)
            assert cond.output_sensi.shape == (n_timepoints, n_par, n_obs)


def test_simple_prediction(edata_objects):
    """
    Test prediction without using PEtab, using default postprocessing first
    """

    # get the model and the edatas, create an objective
    model, solver, edatas = edata_objects
    objective = pypesto.AmiciObjective(model, solver, edatas[0], 1)
    # now create a prediction object
    default_predictor = AmiciPredictor(objective)
    # let's set the parameter vector
    x = np.array([3.0, 0.5])

    # assert output is what it should look like when running in efault mode
    p = default_predictor(x)
    check_outputs(p, out=(0,), n_cond=1, n_timepoints=10, n_obs=2, n_par=2)

    # assert folder is there with all files
    # remove file is already existing
    if os.path.exists("deleteme"):
        shutil.rmtree("deleteme")
    p = default_predictor(
        x, output_file="deleteme.csv", sensi_orders=(1,), output_format="csv"
    )
    check_outputs(p, out=(1,), n_cond=1, n_timepoints=10, n_obs=2, n_par=2)
    # check created files
    assert os.path.exists("deleteme")
    assert set(os.listdir("deleteme")) == {
        "deleteme_0__s0.csv",
        "deleteme_0__s1.csv",
    }
    shutil.rmtree("deleteme")

    # assert h5 file is there
    p = default_predictor(x, output_file="deleteme.h5", output_format="h5")
    check_outputs(p, out=(0,), n_cond=1, n_timepoints=10, n_obs=2, n_par=2)
    assert os.path.exists("deleteme.h5")
    os.remove("deleteme.h5")


def test_complex_prediction(edata_objects):
    """
    Test prediction without using PEtab, using user-defined postprocessing
    """

    def pp_out(raw_outputs):
        # compute ratios of simulations across conditions
        amici_y = [raw_output["y"] for raw_output in raw_outputs]
        outs1 = np.array(
            [
                amici_y[0][:, 1] / amici_y[0][:, 0],
                amici_y[1][:, 1] / amici_y[0][:, 0],
                amici_y[1][:, 1] / amici_y[1][:, 0],
                amici_y[2][:, 1] / amici_y[0][:, 0],
                amici_y[2][:, 1] / amici_y[2][:, 0],
            ]
        ).transpose()
        outs2 = np.array(
            [
                amici_y[0][:, 1] / amici_y[0][:, 0],
                amici_y[0][:, 1] / amici_y[1][:, 0],
                amici_y[1][:, 1] / amici_y[1][:, 0],
                amici_y[2][:, 1] / amici_y[1][:, 0],
                amici_y[2][:, 1] / amici_y[2][:, 0],
            ]
        ).transpose()
        return [outs1, outs2]

    def pps_out(raw_outputs):
        amici_y = [raw_output["y"] for raw_output in raw_outputs]
        amici_sy = [raw_output["sy"] for raw_output in raw_outputs]
        # compute ratios of simulations across conditions (yes, I know this is
        # symbolically wrong, but we only check the shape of the outputs...)
        s_outs1 = np.zeros((10, 2, 5))
        s_outs1[:, :, 0] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[0][:, 0], (2, 1)).transpose()
        )
        s_outs1[:, :, 1] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[0][:, 0], (2, 1)).transpose()
        )
        s_outs1[:, :, 2] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[0][:, 0], (2, 1)).transpose()
        )
        s_outs1[:, :, 3] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[0][:, 0], (2, 1)).transpose()
        )
        s_outs1[:, :, 4] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[0][:, 0], (2, 1)).transpose()
        )

        s_outs2 = np.zeros((10, 2, 5))
        s_outs2[:, :, 0] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[0][:, 0], (2, 1)).transpose()
        )
        s_outs2[:, :, 1] = (
            amici_sy[0][:, 1, :]
            / np.tile(amici_y[1][:, 0], (2, 1)).transpose()
        )
        s_outs2[:, :, 2] = (
            amici_sy[1][:, 1, :]
            / np.tile(amici_y[1][:, 0], (2, 1)).transpose()
        )
        s_outs2[:, :, 3] = (
            amici_sy[2][:, 1, :]
            / np.tile(amici_y[1][:, 0], (2, 1)).transpose()
        )
        s_outs2[:, :, 4] = (
            amici_sy[2][:, 1, :]
            / np.tile(amici_y[2][:, 0], (2, 1)).transpose()
        )
        return [s_outs1, s_outs2]

    def ppt_out(raw_outputs):
        amici_t = [raw_output["t"] for raw_output in raw_outputs]
        # compute ratios of simulations across conditions
        t_out1 = amici_t[0]
        t_out2 = amici_t[1]

        return [t_out1, t_out2]

    # get the model and the edatas, create an objective
    model, solver, edatas = edata_objects
    objective = pypesto.AmiciObjective(model, solver, edatas, 1)
    # now create a prediction object
    complex_predictor = AmiciPredictor(
        objective,
        max_chunk_size=2,
        post_processor=pp_out,
        post_processor_sensi=pps_out,
        post_processor_time=ppt_out,
        output_ids=[f"ratio_{i_obs}" for i_obs in range(5)],
    )
    # let's set the parameter vector
    x = np.array([3.0, 0.5])

    # assert output is what it should look like when running in efault mode
    p = complex_predictor(x, sensi_orders=(0, 1))
    check_outputs(p, out=(0, 1), n_cond=2, n_timepoints=10, n_obs=5, n_par=2)

    # assert folder is there with all files
    # remove file is already existing
    if os.path.exists("deleteme"):
        shutil.rmtree("deleteme")
    p = complex_predictor(
        x, output_file="deleteme.csv", sensi_orders=(0, 1), output_format="csv"
    )
    check_outputs(p, out=(0, 1), n_cond=2, n_timepoints=10, n_obs=5, n_par=2)
    # check created files
    assert os.path.exists("deleteme")
    expected_files = {
        "deleteme_0.csv",
        "deleteme_0__s0.csv",
        "deleteme_0__s1.csv",
        "deleteme_1.csv",
        "deleteme_1__s0.csv",
        "deleteme_1__s1.csv",
    }
    assert set(os.listdir("deleteme")) == expected_files
    shutil.rmtree("deleteme")

    # assert h5 file is there
    p = complex_predictor(
        x, output_file="deleteme.h5", sensi_orders=(0, 1), output_format="h5"
    )
    check_outputs(p, out=(0, 1), n_cond=2, n_timepoints=10, n_obs=5, n_par=2)
    assert os.path.exists("deleteme.h5")
    os.remove("deleteme.h5")


def test_petab_prediction():
    """
    Test prediction via PEtab
    """
    model_name = "conversion_reaction"

    # get the PEtab model
    yaml_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "doc",
        "example",
        model_name,
        f"{model_name}.yaml",
    )
    petab_problem = petab.Problem.from_yaml(yaml_file)
    # import PEtab problem
    petab_problem.model_name = f"{model_name}_petab"
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create prediction via PAteb
    predictor = importer.create_predictor()

    # ===== run test for prediction ===========================================
    p = predictor(
        np.array(petab_problem.x_nominal_free_scaled), sensi_orders=(0, 1)
    )
    check_outputs(p, out=(0, 1), n_cond=1, n_timepoints=10, n_obs=1, n_par=2)
    # check outputs for simulation and measurement dataframes
    importer.prediction_to_petab_measurement_df(p, predictor)
    importer.prediction_to_petab_simulation_df(p, predictor)

    # ===== run test for ensemble prediction ==================================
    # read a set of ensemble vectors from the csv
    ensemble_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "doc",
        "example",
        model_name,
        "parameter_ensemble.tsv",
    )
    ensemble = pypesto.ensemble.read_from_csv(
        ensemble_file,
        lower_bound=petab_problem.get_lb(),
        upper_bound=petab_problem.get_ub(),
    )
    isinstance(ensemble, pypesto.ensemble.Ensemble)

    # check summary creation and identifiability analysis
    summary = ensemble.compute_summary(percentiles_list=[10, 25, 75, 90])
    assert isinstance(summary, dict)
    assert set(summary.keys()) == {
        "mean",
        "std",
        "median",
        "percentile 10",
        "percentile 25",
        "percentile 75",
        "percentile 90",
    }

    parameter_identifiability = ensemble.check_identifiability()
    assert isinstance(parameter_identifiability, pd.DataFrame)

    # perform a prediction for the ensemble
    ensemble_prediction = ensemble.predict(predictor=predictor)
    # check some of the basic functionality: compressing output to large arrays
    ensemble_prediction.condense_to_arrays()
    for field in ("timepoints", "output", "output_sensi"):
        isinstance(ensemble_prediction.prediction_arrays[field], np.ndarray)

    # computing summaries
    ensemble_prediction.compute_summary(percentiles_list=[5, 20, 80, 95])
    isinstance(ensemble_prediction, pypesto.ensemble.EnsemblePrediction)

    # define some short hands
    pred = ensemble_prediction.prediction_summary
    keyset = {
        "mean",
        "std",
        "median",
        "percentile 5",
        "percentile 20",
        "percentile 80",
        "percentile 95",
    }
    # check some properties
    assert set(pred.keys()) == keyset
    for key in keyset:
        assert pred[key].comment == key

    # check some particular properties of this example
    assert pred["mean"].conditions[0].output[0, 0] == 1.0
    assert pred["median"].conditions[0].output[0, 0] == 1.0
    assert pred["std"].conditions[0].output[0, 0] == 0.0

    # check writing to h5
    pypesto.ensemble.write_ensemble_prediction_to_h5(
        ensemble_prediction, "deleteme_ensemble.h5"
    )
    assert os.path.exists("deleteme_ensemble.h5")
    os.remove("deleteme_ensemble.h5")

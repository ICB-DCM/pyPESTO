"""
This is for testing the pypesto.objective.AmiciPrediction.
"""

import amici
import pypesto
import os
import sys
import numpy as np
import shutil
import pytest
from pypesto.objective.amici_prediction import (PredictionResult,
                                                PredictionConditionResult)


def create_testmodel():
    # read in sbml file
    sbml_file = '../../doc/example/conversion_reaction/model_conversion_reaction.xml'
    model_name = 'conversion_reaction'
    model_output_dir = '../../doc/example/tmp/conversion_reaction_enhanced'

    # try to import the exisiting model, if possible
    try:
        sys.path.insert(0, os.path.abspath(model_output_dir))
        model_module = amici.import_model_module(model_name, model_output_dir)
        model = model_module.getModel()
    except:
        # read in and adapt the sbml slightly
        if os.path.abspath(model_output_dir) in sys.path:
            sys.path.remove(os.path.abspath(model_output_dir))
        sbml_importer = amici.SbmlImporter(sbml_file)

        # add observables to sbml model
        def create_observable(sbml_model, obs_id):
            # create a parameter, which will get a rule assignmed as observable
            parameter = sbml_model.createParameter()
            parameter.setId(f'observable_{obs_id}')
            parameter.setName(f'observable_{obs_id}')
            parameter.constant = True

            rule = sbml_importer.sbml.createAssignmentRule()
            rule.setId(f'observable_{obs_id}')
            rule.setName(f'observable_{obs_id}')
            rule.setVariable(f'observable_{obs_id}')
            rule.setFormula(obs_id)

        create_observable(sbml_importer.sbml, 'A')
        create_observable(sbml_importer.sbml, 'B')

        # add constant parameters and observables to AMICI model
        constantParameters = ['A0', 'B0']
        observables = amici.assignmentRules2observables(
            sbml_importer.sbml,  # the libsbml model object
            filter_function=lambda variable:
            variable.getId().startswith('observable_')
        )
        # generate the python module for the model.
        sbml_importer.sbml2amici(model_name,
                                 model_output_dir,
                                 verbose=False,
                                 observables=observables,
                                 constantParameters=constantParameters)

        # Importing the module and loading the model
        sys.path.insert(0, os.path.abspath(model_output_dir))
        model_module = amici.import_model_module(model_name, model_output_dir)
        model = model_module.getModel()

    return model


@pytest.fixture()
def edata_objects():

    testmodel = create_testmodel()

    # set timepoints for which we want to simulate the model
    testmodel.setTimepoints(np.linspace(0, 4, 10))
    testmodel.setParameters(np.array([4., 0.4]))
    # Create solver instance
    solver = testmodel.getSolver()

    # create edatas
    rdatas = []
    edatas = []
    fixedParameters = [np.array([2., 0.]), np.array([0., 4.]),
                       np.array([1., 1.])]
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
            assert cond.output_sensi.shape == (n_timepoints, n_obs, n_par)

def test_simple_prediction(edata_objects):
    """
    Test prediction without using PEtab, using default postprocessing first
    """

    # get the model and the edatas, create an objective
    model, solver, edatas = edata_objects
    objective = pypesto.AmiciObjective(model, solver, edatas[0], 1)
    # now create a prediction object
    default_prediction = pypesto.AmiciPrediction(objective)
    # let's set the parameter vector
    x = np.array([3., 0.5])

    # assert output is what it should look like when running in efault mode
    p = default_prediction(x)
    check_outputs(p, out=(0,), n_cond=1, n_timepoints=10, n_obs=2, n_par=2)

    # assert folder is there with all files
    # remove file is already existing
    if os.path.exists('deleteme'):
        shutil.rmtree('deleteme')
    p = default_prediction(x, output_file='deleteme.csv', sensi_orders=(1,),
                           output_format='csv')
    check_outputs(p, out=(1,), n_cond=1, n_timepoints=10, n_obs=2, n_par=2)
    # check created files
    assert os.path.exists('deleteme')
    assert os.listdir('deleteme') == ['deleteme_0__s0.csv',
                                      'deleteme_0__s1.csv']
    shutil.rmtree('deleteme')

    # assert h5 file is there
    p = default_prediction(x, output_file='deleteme.h5', output_format='h5')
    check_outputs(p, out=(0,), n_cond=1, n_timepoints=10, n_obs=2, n_par=2)
    assert os.path.exists('deleteme.h5')
    os.remove('deleteme.h5')


def test_complex_prediction(edata_objects):
    """
    Test prediction without using PEtab, using user-defined postprocessing
    """

    def pp_out(amici_y):
        # compute ratios of simulations across conditions
        outs1 = np.array([
            amici_y[0][:, 1] / amici_y[0][:, 0],
            amici_y[1][:, 1] / amici_y[0][:, 0],
            amici_y[1][:, 1] / amici_y[1][:, 0],
            amici_y[2][:, 1] / amici_y[0][:, 0],
            amici_y[2][:, 1] / amici_y[2][:, 0],
        ]).transpose()
        outs2 = np.array([
            amici_y[0][:, 1] / amici_y[0][:, 0],
            amici_y[0][:, 1] / amici_y[1][:, 0],
            amici_y[1][:, 1] / amici_y[1][:, 0],
            amici_y[2][:, 1] / amici_y[1][:, 0],
            amici_y[2][:, 1] / amici_y[2][:, 0],
        ]).transpose()
        return [outs1, outs2]

    def pps_out(amici_y, amici_sy):
        # compute ratios of simulations across conditions (yes, I know this is
        # symbolically wrong, but we only check the shape of the outputs atm...)
        s_outs1 = np.zeros((10, 5, 2))
        s_outs1[:,0,:] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[0][:, 0], (2,1)).transpose()
        s_outs1[:,1,:] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[0][:, 0], (2,1)).transpose()
        s_outs1[:,2,:] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[0][:, 0], (2,1)).transpose()
        s_outs1[:,3,:] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[0][:, 0], (2,1)).transpose()
        s_outs1[:,4,:] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[0][:, 0], (2,1)).transpose()

        s_outs2 = np.zeros((10, 5, 2))
        s_outs2[:, 0, :] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[0][:, 0], (2,1)).transpose()
        s_outs2[:, 1, :] = \
            amici_sy[0][:, 1, :] / np.tile(amici_y[1][:, 0], (2,1)).transpose()
        s_outs2[:, 2, :] = \
            amici_sy[1][:, 1, :] / np.tile(amici_y[1][:, 0], (2,1)).transpose()
        s_outs2[:, 3, :] = \
            amici_sy[2][:, 1, :] / np.tile(amici_y[1][:, 0], (2,1)).transpose()
        s_outs2[:, 4, :] = \
            amici_sy[2][:, 1, :] / np.tile(amici_y[2][:, 0], (2,1)).transpose()
        return [s_outs1, s_outs2]

    def ppt_out(amici_t):
        # compute ratios of simulations across conditions
        t_out1 = amici_t[0]
        t_out2 = amici_t[1]

        return [t_out1, t_out2]

    # get the model and the edatas, create an objective
    model, solver, edatas = edata_objects
    objective = pypesto.AmiciObjective(model, solver, edatas, 1)
    # now create a prediction object
    complex_prediction = pypesto.AmiciPrediction(objective,
        max_num_conditions=2, post_processing=pp_out,
        post_processing_sensi=pps_out, post_processing_timepoints=ppt_out,
        observables=[f'ratio_{i_obs}' for i_obs in range(5)])
    # let's set the parameter vector
    x = np.array([3., 0.5])

    # assert output is what it should look like when running in efault mode
    p = complex_prediction(x, sensi_orders=(0,1))
    check_outputs(p, out=(0,1), n_cond=2, n_timepoints=10, n_obs=5, n_par=2)

    # assert folder is there with all files
    # remove file is already existing
    if os.path.exists('deleteme'):
        shutil.rmtree('deleteme')
    p = complex_prediction(x, output_file='deleteme.csv', sensi_orders=(0,1),
                           output_format='csv')
    check_outputs(p, out=(0,1), n_cond=2, n_timepoints=10, n_obs=5, n_par=2)
    # check created files
    assert os.path.exists('deleteme')
    expected_files = {'deleteme_0.csv', 'deleteme_0__s0.csv',
                      'deleteme_0__s1.csv', 'deleteme_1.csv',
                      'deleteme_1__s0.csv', 'deleteme_1__s1.csv'}
    assert set(os.listdir('deleteme')) == expected_files
    shutil.rmtree('deleteme')

    # assert h5 file is there
    p = complex_prediction(x, output_file='deleteme.h5', sensi_orders=(0,1),
                           output_format='h5')
    check_outputs(p, out=(0,1), n_cond=2, n_timepoints=10, n_obs=5, n_par=2)
    assert os.path.exists('deleteme.h5')
    os.remove('deleteme.h5')

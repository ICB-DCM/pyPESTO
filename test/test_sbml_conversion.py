import unittest
import amici
import pesto
import pesto.optimize.optimizer
import pesto.optimize.optimize
import importlib
import os
import sys
import numpy as np
import statistics
import warnings

optimizers = {
    'scipy': ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',],
    # disabled: 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
    'dlib' : ['global']
}

class OptimizerTest(unittest.TestCase):
    def runTest(self):
        for example in ['conversion_reaction']:
            objective, model = load_model_objective(example)
            target_fval = objective.get_fval(list(model.getParameters()))
            for library in ['scipy','dlib']:
                for method in optimizers[library]:
                    with self.subTest(library=library, caseName=method):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            test_parameter_estimation(objective, model, library + '_' + method, 20, target_fval)


def test_parameter_estimation(objective, model, solver, n_starts, target_fval):
    optimizer = pesto.optimize.optimizer.Optimizer(solver=solver)

    problem = pesto.problem.Problem(objective, model)
    problem.generate_starting_points(n_starts)

    results = pesto.optimize.optimize.optimize(problem, optimizer, result=None)

    successes = [result for result in results if result.fun < target_fval]

    summary = solver + ':\n ' + str(len(successes)) + '/' + str(len(results)) + ' reached target\n'
    if 'nfev' in dir(results[0]):
        function_evals = [result.nfev for result in successes]
        summary = summary + 'mean fun evals:' + str(statistics.mean(function_evals)) \
                  + '±' + str(statistics.stdev(function_evals)/n_starts) + '\n'

    if 'njev' in dir(results[0]):
        grad_evals = [result.njev for result in successes]
        summary = summary + 'mean grad evals:' + str(statistics.mean(grad_evals)) \
                  + '±' + str(statistics.stdev(grad_evals)/n_starts) + '\n'

    print(summary)

    assert(len(successes))


def load_model_objective(example_name):
    sbml_file = os.path.join('example', 'model_' + example_name + '.xml')
    # name of the model that will also be the name of the python module
    model_name = 'model_' + example_name
    # directory to which the generated model code is written
    model_output_dir = os.path.join('example', example_name )

    # import sbml model, complile and generate amici module
    sbml_importer = amici.SbmlImporter(sbml_file)
    sbml_importer.sbml2amici(model_name,
                             model_output_dir,
                             verbose=False)

    # load amici module (the usual starting point later for the analysis)
    sys.path.insert(0, os.path.abspath(model_output_dir))
    model_module = importlib.import_module(model_name)
    model = model_module.getModel()
    model.requireSensitivitiesForAllParameters()
    model.setTimepoints(amici.DoubleVector(np.linspace(0, 10, 11)))
    model.setParameterScale(amici.AMICI_SCALING_LOG10)
    model.setParameters(amici.DoubleVector([-0.3, -0.7]))
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.AMICI_SENSI_FSA)
    solver.setSensitivityOrder(amici.AMICI_SENSI_ORDER_FIRST)

    # generate experimental data
    rdata = amici.runAmiciSimulation(model, solver, None)
    edata = amici.ExpData(rdata['ptr'].get(), 0.05, 0.0)

    return pesto.objective.AmiciObjective(model, solver, [edata], 1), model


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(OptimizerTest())
    unittest.main()

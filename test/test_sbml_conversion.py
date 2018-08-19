import os
import sys
import unittest
import amici
import pesto
import importlib
import numpy as np
import statistics
import warnings


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

optimizers = {
    'scipy': ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
              'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
              'trust-ncg', 'trust-exact', 'trust-krylov',
              'ls_trf', 'ls_dogbox'],
    # disabled: ,'trust-constr', 'ls_lm', 'dogleg'
    'dlib': ['default']
}


class OptimizerTest(unittest.TestCase):
    def runTest(self):
        for example in ['conversion_reaction']:
            objective, model = _load_model_objective(example)
            target_fval = objective.get_fval(list(model.getParameters()))
            for library in optimizers.keys():
                for method in optimizers[library]:
                    with self.subTest(library=library, solver=method):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            _test_parameter_estimation(objective,
                                                       library,
                                                       method,
                                                       25,
                                                       target_fval)


def _test_parameter_estimation(objective, library, solver, n_starts,
                              target_fval):

    options = {
        'maxiter': 100
    }

    if library == 'scipy':
        optimizer = pesto.optimize.optimizer.ScipyOptimizer(method=solver,
                                                            options=options)
    elif library == 'dlib':
        optimizer = pesto.optimize.optimizer.DlibOptimizer(method=solver,
                                                           options=options)

    lb = -2 * np.ones((1, objective.dim))
    ub = 2 * np.ones((1, objective.dim))
    problem = pesto.problem.Problem(objective, lb, ub)

    results = pesto.optimize.minimize(
        problem, optimizer, n_starts,
        startpoint_method=pesto.optimize.startpoint.uniform)
    results = results.optimizer_results

    successes = [result for result in results if result.fval < target_fval]

    summary = solver + ':\n ' + str(len(successes)) \
        + '/' + str(len(results)) + ' reached target\n'

    if hasattr(results[0], 'n_fval'):
        function_evals = [result.n_fval for result in results if result.n_fval]
        if len(function_evals):
            summary = summary + 'mean fun evals:' \
                + str(statistics.mean(function_evals)) \
                + '±' + str(statistics.stdev(function_evals) / n_starts) + '\n'

    if hasattr(results[0], 'n_grad'):
        grad_evals = [result.n_grad for result in results if result.n_grad]
        if len(grad_evals):
            summary = summary + 'mean grad evals:' \
                + str(statistics.mean(grad_evals)) \
                + '±' + str(statistics.stdev(grad_evals) / n_starts) + '\n'

    if hasattr(results[0], 'n_hess'):
        hess_evals = [result.n_hess for result in results if result.n_hess]
        if len(hess_evals):
            summary = summary + 'mean hess evals:' \
                + str(statistics.mean(hess_evals)) \
                + '±' + str(statistics.stdev(hess_evals) / n_starts) + '\n'

    print(summary)

    assert(len(successes))


def _load_model_objective(example_name):
    sbml_file = os.path.join('example', 'model_' + example_name + '.xml')
    # name of the model that will also be the name of the python module
    model_name = 'model_' + example_name
    # directory to which the generated model code is written
    model_output_dir = os.path.join('example', example_name)

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
    model.setParameterScale(amici.ParameterScaling_log10)
    model.setParameters(amici.DoubleVector([-0.3, -0.7]))
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.SensitivityMethod_forward)
    solver.setSensitivityOrder(amici.SensitivityOrder_first)

    # generate experimental data
    rdata = amici.runAmiciSimulation(model, solver, None)
    edata = amici.ExpData(rdata['ptr'].get(), 0.05, 0.0)

    return pesto.objective.AmiciObjective(model, solver, [edata], 2), model


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(OptimizerTest())
    unittest.main()

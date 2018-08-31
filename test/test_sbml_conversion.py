import os
import sys
import unittest
import amici
import pypesto
import importlib
import numpy as np
import statistics
import warnings


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

optimizers = {
    'scipy': ['Powell', 'trust-exact', 'trust-krylov',
              'ls_trf', 'ls_dogbox'],
    # disabled: ,'trust-constr', 'ls_lm', 'dogleg'
    'dlib': ['default']
}


class AmiciObjectiveTest(unittest.TestCase):
    def runTest(self):
        for example in ['conversion_reaction']:
            objective, model = _load_model_objective(example)
            x0 = list(model.getParameters())
            df = objective.check_grad(
                x0,
                eps=1e-5,
                verbosity=0,
                mode='MODE_FUN'
            )
            self.assertTrue(np.all(df.rel_err.values < 1e-3))
            self.assertTrue(np.all(df.abs_err.values < 1e-1))
            df = objective.check_grad(
                x0,
                eps=1e-5,
                verbosity=0,
                mode='MODE_RES'
            )
            self.assertTrue(np.all(df.rel_err.values < 1e-6))
            self.assertTrue(np.all(df.abs_err.values < 1e-6))

            target_fval = objective.get_fval(list(model.getParameters()))
            for library in optimizers.keys():
                for method in optimizers[library]:
                    with self.subTest(library=library, solver=method):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.parameter_estimation(
                                objective,
                                library,
                                method,
                                1,
                                target_fval)

    def parameter_estimation(
        self,
        objective,
        library,
        solver,
        n_starts,
        target
    ):
        options = {
            'maxiter': 100
        }

        if library == 'scipy':
            optimizer = pypesto.ScipyOptimizer(method=solver,
                                               options=options)
        elif library == 'dlib':
            optimizer = pypesto.DlibOptimizer(method=solver,
                                              options=options)

        optimizer.temp_file = os.path.join('test', 'tmp_{index}.csv')

        lb = -2 * np.ones((1, objective.dim))
        ub = 2 * np.ones((1, objective.dim))
        problem = pypesto.Problem(objective, lb, ub)

        results = pypesto.minimize(
            problem, optimizer, n_starts,
            startpoint_method=pypesto.optimize.startpoint.uniform,
            allow_failed_starts=False)
        results = results.optimize_result.list

        successes = [result for result in results if result.fval < target]

        summary = solver + ':\n ' + str(len(successes)) \
            + '/' + str(len(results)) + ' reached target\n'

        function_evals = [result.n_fval for result in results]
        if len(function_evals):
            summary = summary + 'mean fun evals:' \
                + str(statistics.mean(function_evals)) \
                + '±' + str(statistics.stdev(function_evals) / n_starts) \
                + '\n'

        grad_evals = [result.n_grad for result in results]
        if len(grad_evals):
            summary = summary + 'mean grad evals:' \
                + str(statistics.mean(grad_evals)) \
                + '±' + str(statistics.stdev(grad_evals) / n_starts) \
                + '\n'

        hess_evals = [result.n_hess for result in results]
        if len(hess_evals):
            summary = summary + 'mean hess evals:' \
                + str(statistics.mean(hess_evals)) \
                + '±' + str(statistics.stdev(hess_evals) / n_starts) \
                + '\n'

        print(summary)

        #self.assertTrue(len(successes) > 0)


def _load_model_objective(example_name):
    sbml_file = os.path.join('doc', 'example',
                             'model_' + example_name + '.xml')
    # name of the model that will also be the name of the python module
    model_name = 'model_' + example_name
    # directory to which the generated model code is written
    model_output_dir = os.path.join('doc', 'example',
                                    example_name)

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

    return pypesto.AmiciObjective(model, solver, [edata], 2), model


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AmiciObjectiveTest())
    unittest.main()

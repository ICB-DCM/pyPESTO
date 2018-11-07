import os
import sys
import unittest
import amici
import pypesto
import importlib
import numpy as np
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


class AmiciObjectiveTest(unittest.TestCase):
    def runTest(self):
        for example in ['conversion_reaction']:
            objective, model = _load_model_objective(example)
            x0 = list(model.getParameters())
            df = objective.check_grad(
                x0,
                eps=1e-5,
                verbosity=0,
                mode=pypesto.objective.constants.MODE_FUN
            )
            self.assertTrue(np.all(df.rel_err.values < 1e-2))
            self.assertTrue(np.all(df.abs_err.values < 1e-1))
            df = objective.check_grad(
                x0,
                eps=1e-5,
                verbosity=0,
                mode=pypesto.objective.constants.MODE_RES
            )
            self.assertTrue(np.all(df.rel_err.values < 1e-6))
            self.assertTrue(np.all(df.abs_err.values < 1e-6))

            for library in optimizers.keys():
                for method in optimizers[library]:
                    for fp in [[], [1]]:
                        with self.subTest(library=library,
                                          solver=method,
                                          fp=fp):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                parameter_estimation(
                                    objective,
                                    library,
                                    method,
                                    fp,
                                    1)


def parameter_estimation(
    objective,
    library,
    solver,
    fixed_pars,
    n_starts,
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
    pars = objective.amici_model.getParameters()
    problem = pypesto.Problem(objective, lb, ub,
                              x_fixed_indices=fixed_pars,
                              x_fixed_vals=[pars[idx] for idx in fixed_pars]
                              )

    optimize_options = pypesto.OptimizeOptions(allow_failed_starts=False)

    results = pypesto.minimize(
        problem, optimizer, n_starts, options=optimize_options)
    results = results.optimize_result.list


def _load_model_objective(example_name):
    # name of the model that will also be the name of the python module
    model_name = 'model_' + example_name

    # sbml file
    sbml_file = os.path.join('doc', 'example', example_name,
                             model_name + '.xml')

    # directory to which the generated model code is written
    model_output_dir = os.path.join('doc', 'example', 'tmp',
                                    model_name)

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
    model.setTimepoints(np.linspace(0, 10, 11))
    model.setParameterScale(amici.ParameterScaling_log10)
    model.setParameters([-0.3, -0.7])
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.SensitivityMethod_forward)
    solver.setSensitivityOrder(amici.SensitivityOrder_first)

    # generate experimental data
    rdata = amici.runAmiciSimulation(model, solver, None)
    edata = amici.ExpData(rdata, 0.05, 0.0)

    options = pypesto.objective.ObjectiveOptions(
        trace_record=True,
        trace_record_hess=False,
        trace_all=True,
        trace_file='tmp/traces/conversion_example_{index}.csv',
        trace_save_iter=1
    )

    return (pypesto.AmiciObjective(model, solver, [edata], 2, options=options),
            model)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AmiciObjectiveTest())
    unittest.main()

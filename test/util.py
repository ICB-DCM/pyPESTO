"""Various test problems and utility functions."""


import os
import sys
import numpy as np
import scipy.optimize as so
import importlib
import pypesto

try:
    import amici
except ImportError:
    pass


def obj_for_sensi(fun, grad, hess, max_sensi_order, integrated, x):
    """
    Create a pypesto.Objective able to compute up to the speficied
    max_sensi_order. Returns a dict containing the objective obj as well
    as max_sensi_order and fval, grad, hess for the passed x.

    Parameters
    ----------

    fun, grad, hess: callable
        Functions computing the fval, grad, hess.
    max_sensi_order: int
        Maximum sensitivity order the pypesto.Objective should be capable of.
    integrated: bool
        True if fun, grad, hess should be integrated into one function, or
        passed to pypesto.Objective separately (both is possible)
    x: np.array
        Value at which to evaluate the function to obtain true values.

    Returns
    -------

    ret: dict
        With fields obj, max_sensi_order, x, fval, grad, hess.
    """
    if integrated:
        if max_sensi_order == 2:
            def arg_fun(x):
                return fun(x), grad(x), hess(x)
            arg_grad = arg_hess = True
        elif max_sensi_order == 1:
            def arg_fun(x):
                return fun(x), grad(x)
            arg_grad = True
            arg_hess = False
        else:
            def arg_fun(x):
                return fun(x)
            arg_grad = arg_hess = False
    else:  # integrated
        if max_sensi_order >= 2:
            arg_hess = hess
        else:
            arg_hess = None
        if max_sensi_order >= 1:
            arg_grad = grad
        else:
            arg_grad = None
        arg_fun = fun
    obj = pypesto.Objective(fun=arg_fun, grad=arg_grad, hess=arg_hess)
    return {'obj': obj,
            'max_sensi_order': max_sensi_order,
            'x': x,
            'fval': fun(x),
            'grad': grad(x),
            'hess': hess(x)}


def rosen_for_sensi(max_sensi_order, integrated=False, x=None):
    """
    Rosenbrock function from scipy.optimize.
    """
    if x is None:
        x = [0, 1]

    return obj_for_sensi(so.rosen,
                         so.rosen_der,
                         so.rosen_hess,
                         max_sensi_order, integrated, x)


def poly_for_sensi(max_sensi_order, integrated=False, x=0.):
    """
    1-dim polynomial for testing in 1d.
    """

    def fun(x):
        return (x - 2)**2 + 1

    def grad(x):
        return 2 * (x - 2)

    def hess(_):
        return 2

    return obj_for_sensi(fun, grad, hess,
                         max_sensi_order, integrated, x)


def load_amici_objective(example_name):
    # name of the model that will also be the name of the python module
    model_name = 'model_' + example_name

    # sbml file
    sbml_file = os.path.join('doc', 'example', example_name,
                             model_name + '.xml')

    # directory to which the generated model code is written
    model_output_dir = os.path.join('doc', 'example', 'tmp',
                                    model_name)

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    sys.path.insert(0, os.path.abspath(model_output_dir))

    try:
        model_module = importlib.import_module(model_name)
        model = model_module.getModel()
    except ModuleNotFoundError:
        # import sbml model, compile and generate amici module
        sbml_importer = amici.SbmlImporter(sbml_file)
        sbml_importer.sbml2amici(model_name,
                                 model_output_dir,
                                 verbose=False)
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

    return (pypesto.AmiciObjective(model, solver, [edata], 2),
            model)

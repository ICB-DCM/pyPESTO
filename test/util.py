"""Various test problems and utility functions."""


import os
import sys
import numpy as np
import scipy.optimize as so
import importlib
import autograd.numpy as anp
from autograd import jacobian
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

            def arg_res(x):
                return grad(x), hess(x)

            arg_sres = True
        elif max_sensi_order == 1:

            def arg_fun(x):
                return fun(x), grad(x)

            arg_grad = True
            arg_hess = False

            def arg_res(x):
                return grad(x)

            arg_sres = False
        else:

            def arg_fun(x):
                return fun(x)

            arg_grad = arg_hess = False
            arg_res = arg_sres = False
    else:  # integrated
        if max_sensi_order >= 2:
            arg_hess = hess
            arg_sres = hess
        else:
            arg_hess = None
            arg_sres = None
        if max_sensi_order >= 1:
            arg_grad = grad
            arg_res = grad
        else:
            arg_grad = None
            arg_res = None
        arg_fun = fun
    obj = pypesto.Objective(
        fun=arg_fun, grad=arg_grad, hess=arg_hess, res=arg_res, sres=arg_sres
    )
    return {
        "obj": obj,
        "max_sensi_order": max_sensi_order,
        "x": x,
        "fval": fun(x),
        "grad": grad(x),
        "hess": hess(x),
    }


def rosen_for_sensi(max_sensi_order, integrated=False, x=None):
    """
    Rosenbrock function from scipy.optimize.
    """
    if x is None:
        x = [0, 1]

    return obj_for_sensi(
        so.rosen, so.rosen_der, so.rosen_hess, max_sensi_order, integrated, x
    )


def poly_for_sensi(max_sensi_order, integrated=False, x=0.0):
    """
    1-dim polynomial for testing in 1d.
    """

    def fun(x):
        return (x - 2) ** 2 + 1

    def grad(x):
        return 2 * (x - 2)

    def hess(_):
        return 2

    return obj_for_sensi(fun, grad, hess, max_sensi_order, integrated, x)


class CRProblem:
    """ODE model of a conversion reaction x0 <-> x1.

    Parameters: reaction rate coefficients p0: x0 -> x1, p1: x1 -> x0.
    Translates to the ODE

    .. math::

        \\frac{dx_0}{dt} = -p_0 \\cdot x_0 + p_1 \\cdot x_1,\\
        \\frac{dx_1}{dt} = p_0 \\cdot x_0 - p_1 \\cdot x_1

    Uses automatic differentiation for derivative calculation.
    """

    def __init__(
        self,
        n_t: int = 10,
        max_t: float = 15.0,
        x0: anp.ndarray = None,
        p_true: anp.ndarray = None,
        sigma: float = 0.02,
        lb: anp.ndarray = None,
        ub: anp.ndarray = None,
    ):
        """
        Parameters
        ----------
        n_t: Number of time points.
        max_t: Maximum time point value.
        x0: Initial state.
        p_true: True parameter value.
        sigma: Standard deviation of a normal noise model.
        lb: Lower bound.
        ub: Upper bound.
        """
        self.ts = anp.linspace(0, max_t, n_t)

        if x0 is None:
            x0 = anp.array([1.0, 0.0])
        self.x0: anp.ndarray = x0

        if p_true is None:
            p_true = anp.array([0.06, 0.08])
        self.p_true: anp.ndarray = p_true

        self.sigma: float = sigma

        if lb is None:
            lb = anp.array([0.0, 0.0])
        self.lb = lb

        if ub is None:
            ub = anp.array([0.5, 0.5])
        self.ub = ub

        y_true = self.get_fy()(self.p_true)
        # seed random number generator for reproducibility
        rng = anp.random.Generator(anp.random.PCG64(0))
        self.data = y_true + sigma * rng.normal(size=y_true.shape)

    def get_fy(self):
        """System states, fully observed, analytic solution."""

        def fy(p):
            p0, p1 = p
            e = anp.exp(-(p0 + p1) * self.ts)
            x = (
                1
                / (-p0 - p1)
                * anp.array(
                    [
                        [-p1 - p0 * e, -p1 + p1 * e],
                        [-p0 + p0 * e, -p0 - p1 * e],
                    ]
                )
            )
            y = anp.einsum("mnr,n->mr", x, self.x0)
            return y

        return fy

    def get_fres(self):
        """Residuals."""

        def fres(p):
            return ((self.get_fy()(p) - self.data) / self.sigma).flatten()

        return fres

    def get_fsres(self):
        """Residual sensitivities"""
        return jacobian(self.get_fres())

    def get_ffim(self):
        """Fisher information matrix."""

        def ffim(p):
            sres = self.get_fsres()(p)
            return np.dot(sres.T, sres)

        return ffim

    def get_fnllh(self):
        """Negative log-likelihood (minimization function)."""

        def fnllh(p):
            return 0.5 * anp.sum(self.get_fres()(p) ** 2)

        return fnllh

    def get_fsnllh(self):
        """Negative log-likelihood gradient."""
        return jacobian(self.get_fnllh())

    def get_fs2nllh(self):
        """Negative log-likelihood Hessian."""
        return jacobian(self.get_fsnllh())

    def get_objective(
        self,
        fun: bool = True,
        res: bool = True,
        max_sensi_order: int = 2,
        fim_for_hess: bool = False,
    ):
        """Full pyPESTO objective function.

        Parameters
        ----------
        fun: Whether the objective can calculate function values.
        res: Whether the objective can calculate residuals.
        max_sensi_order: Maximum sensitivity order the function can calculate.
        fim_for_hess: Whether to use the FIM instead of the Hessian.
        """
        if fim_for_hess:
            fhess = self.get_ffim()
        else:
            fhess = self.get_fs2nllh()

        return pypesto.Objective(
            fun=self.get_fnllh() if fun else None,
            grad=self.get_fsnllh() if fun and max_sensi_order >= 1 else None,
            hess=fhess if fun and max_sensi_order >= 2 else None,
            res=self.get_fres() if res else None,
            sres=self.get_fsres() if res and max_sensi_order >= 1 else None,
        )

    def get_problem(self):
        """Full pypesto problem."""
        return pypesto.Problem(
            objective=self.get_objective(),
            lb=self.lb,
            ub=self.ub,
        )


def load_amici_objective(example_name):
    # name of the model that will also be the name of the python module
    model_name = "model_" + example_name

    # sbml file
    sbml_file = os.path.join(
        "doc", "example", example_name, model_name + ".xml"
    )

    # directory to which the generated model code is written
    model_output_dir = os.path.join("doc", "example", "tmp", model_name)

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    sys.path.insert(0, os.path.abspath(model_output_dir))

    try:
        model_module = importlib.import_module(model_name)
        model = model_module.getModel()
    except ModuleNotFoundError:
        # import sbml model, compile and generate amici module
        sbml_importer = amici.SbmlImporter(sbml_file)
        sbml_importer.sbml2amici(model_name, model_output_dir, verbose=False)
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

    return (pypesto.AmiciObjective(model, solver, [edata], 2), model)

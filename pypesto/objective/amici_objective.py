import numpy as np
import copy
import logging
import numbers
from typing import Dict, Tuple, Sequence, Union
from collections import OrderedDict

from .objective import Objective
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS

try:
    import amici
    import amici.petab_objective
    import amici.parameter_mapping
    from amici.parameter_mapping import (
        ParameterMapping, ParameterMappingForCondition)
except ImportError:
    pass

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']

logger = logging.getLogger(__name__)


class AmiciObjective(Objective):
    """
    This class allows to create an objective directly from an amici model.
    """

    def __init__(self,
                 amici_model: AmiciModel,
                 amici_solver: AmiciSolver,
                 edatas: Union[Sequence['amici.ExpData'], 'amici.ExpData'],
                 max_sensi_order: int = None,
                 x_ids: Sequence[str] = None,
                 x_names: Sequence[str] = None,
                 parameter_mapping: 'ParameterMapping' = None,
                 guess_steadystate: bool = True,
                 n_threads: int = 1):
        """
        Constructor.

        Parameters
        ----------

        amici_model:
            The amici model.
        amici_solver:
            The solver to use for the numeric integration of the model.
        edatas:
            The experimental data. If a list is passed, its entries correspond
            to multiple experimental conditions.
        max_sensi_order:
            Maximum sensitivity order supported by the model. Defaults to 2 if
            the model was compiled with o2mode, otherwise 1.
        x_ids:
            Ids of optimization parameters. In the simplest case, this will be
            the AMICI model parameters (default).
        x_names:
            Names of optimization parameters.
        parameter_mapping:
            Mapping of optimization parameters to model parameters. Format
            as created by `amici.petab_objective.create_parameter_mapping`.
            The default is just to assume that optimization and simulation
            parameters coincide.
        guess_steadystate:
            Whether to guess steadystates based on previous steadystates and
            respective derivatives. This option may lead to unexpected
            results for models with conservation laws and should accordingly
            be deactivated for those models.
        n_threads:
            Number of threads that are used for parallelization over
            experimental conditions. If amici was not installed with openMP
            support this option will have no effect.
        """
        if amici is None:
            raise ImportError(
                "This objective requires an installation of amici "
                "(https://github.com/icb-dcm/amici). "
                "Install via `pip3 install amici`.")

        if max_sensi_order is None:
            # 2 if model was compiled with second orders,
            # otherwise 1 can be guaranteed
            max_sensi_order = 2 if amici_model.o2mode else 1

        fun = self.get_bound_fun()

        if max_sensi_order > 0:
            grad = True
            hess = True
        else:
            grad = None
            hess = None

        res = self.get_bound_res()

        if max_sensi_order > 0:
            sres = True
        else:
            sres = None

        super().__init__(
            fun=fun, grad=grad, hess=hess, hessp=None,
            res=res, sres=sres,
            fun_accept_sensi_orders=True,
            res_accept_sensi_orders=True,
        )

        self.amici_model = amici.ModelPtr(amici_model.clone())
        self.amici_solver = amici.SolverPtr(amici_solver.clone())

        # make sure the edatas are a list of edata objects
        if isinstance(edatas, amici.amici.ExpData):
            edatas = [edatas]

        # set the experimental data container
        self.edatas = edatas

        # set the maximum sensitivity order
        self.max_sensi_order = max_sensi_order

        self.guess_steadystate = guess_steadystate

        # optimization parameter ids
        if x_ids is None:
            # use model parameter ids as ids
            x_ids = list(self.amici_model.getParameterIds())
        self.x_ids = x_ids

        self.dim = len(self.x_ids)

        # mapping of parameters
        if parameter_mapping is None:
            # use identity mapping for each condition
            parameter_mapping = create_identity_parameter_mapping(
                amici_model, len(edatas))
        self.parameter_mapping = parameter_mapping

        # preallocate guesses, construct a dict for every edata for which we
        # need to do preequilibration
        if self.guess_steadystate:
            if self.amici_model.ncl() > 0:
                raise ValueError('Steadystate prediciton is not supported for'
                                 'models with conservation laws!')

            if self.amici_model.getSteadyStateSensitivityMode() == \
                    amici.SteadyStateSensitivityMode_simulationFSA:
                raise ValueError('Steadystate guesses cannot be enabled when'
                                 ' `simulationFSA` as '
                                 'SteadyStateSensitivityMode!')
            self.steadystate_guesses = {
                'fval': np.inf,
                'data': {
                    iexp: dict()
                    for iexp, edata in enumerate(self.edatas)
                    if len(edata.fixedParametersPreequilibration) or
                    self.amici_solver.getPreequilibration()
                }
            }
        # optimization parameter names
        if x_names is None:
            # use ids as names
            x_names = x_ids
        self.x_names = x_names

        self.n_threads = n_threads

    def get_bound_fun(self):
        """
        Generate a fun function that calls _call_amici with MODE_FUN. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def fun(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_FUN)

        return fun

    def get_bound_res(self):
        """
        Generate a res function that calls _call_amici with MODE_RES. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def res(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_RES)

        return res

    def rebind_fun(self):
        """
        Replace the current fun function with one that is bound to the current
        instance
        """
        self.fun = self.get_bound_fun()

    def rebind_res(self):
        """
        Replace the current res function with one that is bound to the current
        instance
        """
        self.res = self.get_bound_res()

    def __deepcopy__(self, memodict: Dict = None) -> 'AmiciObjective':
        other = self.__class__.__new__(self.__class__)

        for key in set(self.__dict__.keys()) - \
                {'amici_model', 'amici_solver', 'edatas'}:
            other.__dict__[key] = copy.deepcopy(self.__dict__[key])

        # copy objects that do not have __deepcopy__
        other.amici_model = amici.ModelPtr(self.amici_model.clone())
        other.amici_solver = amici.SolverPtr(self.amici_solver.clone())
        other.edatas = [amici.ExpData(data) for data in self.edatas]

        # rebind functions for __call__
        other.rebind_fun()
        other.rebind_res()

        return other

    def _call_amici(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str
    ) -> Dict:
        # amici is built such that only the maximum sensitivity is required,
        # the lower orders are then automatically computed
        sensi_order = min(max(sensi_orders), 1)
        # order 2 currently not implemented, we are using the FIM

        # check if the requested sensitivities can be computed
        if sensi_order > self.max_sensi_order:
            raise Exception("Sensitivity order not allowed.")

        sensi_method = self.amici_solver.getSensitivityMethod()

        # prepare outputs
        nllh = 0.0
        snllh = np.zeros(self.dim)
        s2nllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        # set order in solver
        self.amici_solver.setSensitivityOrder(sensi_order)

        x_dct = self.par_arr_to_dct(x)

        # fill in parameters
        # TODO (#226) use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=self.edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=self.parameter_mapping,
            amici_model=self.amici_model
        )

        # update steady state
        for data_ix, edata in enumerate(self.edatas):
            if self.guess_steadystate and \
                    self.steadystate_guesses['fval'] < np.inf:
                self.apply_steadystate_guess(data_ix, x_dct)

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            self.amici_model,
            self.amici_solver,
            self.edatas,
            num_threads=min(self.n_threads, len(self.edatas)),
        )

        par_sim_ids = list(self.amici_model.getParameterIds())

        for data_ix, rdata in enumerate(rdatas):
            log_simulation(data_ix, rdata)

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(rdatas)

            condition_map_sim_var = \
                self.parameter_mapping[data_ix].map_sim_var

            nllh -= rdata['llh']

            # compute objective
            if mode == MODE_FUN:

                if sensi_order > 0:
                    add_sim_grad_to_opt_grad(
                        self.x_ids,
                        par_sim_ids,
                        condition_map_sim_var,
                        rdata['sllh'],
                        snllh,
                        coefficient=-1.0
                    )
                    if sensi_method == 1:
                        # TODO Compute the full Hessian, and check here
                        add_sim_hess_to_opt_hess(
                            self.x_ids,
                            par_sim_ids,
                            condition_map_sim_var,
                            rdata['FIM'],
                            s2nllh,
                            coefficient=+1.0
                        )

            elif mode == MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    opt_sres = sim_sres_to_opt_sres(
                        self.x_ids,
                        par_sim_ids,
                        condition_map_sim_var,
                        rdata['sres'],
                        coefficient=1.0
                    )
                    sres = np.vstack([sres, opt_sres]) \
                        if sres.size else opt_sres

        # check whether we should update data for preequilibration guesses
        if self.guess_steadystate and \
                nllh <= self.steadystate_guesses['fval']:
            self.steadystate_guesses['fval'] = nllh
            for data_ix, rdata in enumerate(rdatas):
                self.store_steadystate_guess(data_ix, x_dct, rdata)

        return {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }

    def par_arr_to_dct(self, x: Sequence[float]) -> Dict[str, float]:
        """Create dict from parameter vector."""
        return OrderedDict(zip(self.x_ids, x))

    def get_error_output(self, rdatas: Sequence['amici.ReturnData']):
        """Default output upon error."""
        if not self.amici_model.nt():
            nt = sum([data.nt() for data in self.edatas])
        else:
            nt = sum([data.nt() if data.nt() else self.amici_model.nt()
                      for data in self.edatas])
        n_res = nt * self.amici_model.nytrue

        return {
            FVAL: np.inf,
            GRAD: np.nan * np.ones(self.dim),
            HESS: np.nan * np.ones([self.dim, self.dim]),
            RES:  np.nan * np.ones(n_res),
            SRES: np.nan * np.ones([n_res, self.dim]),
            RDATAS: rdatas
        }

    def apply_steadystate_guess(self, condition_ix: int, x_dct: Dict):
        """
        Use the stored steadystate as well as the respective  sensitivity (
        if available) and parameter value to approximate the steadystate at
        the current parameters using a zeroth or first order taylor
        approximation:
        x_ss(x') = x_ss(x) [+ dx_ss/dx(x)*(x'-x)]
        """
        mapping = self.parameter_mapping[condition_ix].map_sim_var
        x_sim = map_par_opt_to_par_sim(mapping, x_dct, self.amici_model)
        x_ss_guess = []  # resets initial state by default
        if condition_ix in self.steadystate_guesses['data']:
            guess_data = self.steadystate_guesses['data'][condition_ix]
            if guess_data['x_ss'] is not None:
                x_ss_guess = guess_data['x_ss']
            if guess_data['sx_ss'] is not None:
                linear_update = guess_data['sx_ss'].transpose().dot(
                    (x_sim - guess_data['x'])
                )
                # limit linear updates to max 20 % elementwise change
                if (x_ss_guess/linear_update).max() < 0.2:
                    x_ss_guess += linear_update

        self.edatas[condition_ix].x0 = tuple(x_ss_guess)

    def store_steadystate_guess(
            self, condition_ix: int, x_dct: Dict, rdata: 'amici.ReturnData'):
        """
        Store condition parameter, steadystate and steadystate sensitivity in
        steadystate_guesses if steadystate guesses are enabled for this
        condition
        """
        if condition_ix not in self.steadystate_guesses['data']:
            return
        preeq_guesses = self.steadystate_guesses['data'][condition_ix]

        # update parameter
        condition_map_sim_var = \
            self.parameter_mapping[condition_ix].map_sim_var
        x_sim = map_par_opt_to_par_sim(
            condition_map_sim_var, x_dct, self.amici_model)
        preeq_guesses['x'] = x_sim

        # update steadystates
        preeq_guesses['x_ss'] = rdata['x_ss']
        preeq_guesses['sx_ss'] = rdata['sx_ss']

    def reset_steadystate_guesses(self):
        """Resets all steadystate guess data"""
        if not self.guess_steadystate:
            return

        self.steadystate_guesses['fval'] = np.inf
        for condition in self.steadystate_guesses['data']:
            self.steadystate_guesses['data'][condition] = dict()


def log_simulation(data_ix, rdata):
    """Log the simulation results."""
    logger.debug(f"=== DATASET {data_ix} ===")
    logger.debug(f"status: {rdata['status']}")
    logger.debug(f"llh: {rdata['llh']}")

    t_steadystate = 't_steadystate'
    if t_steadystate in rdata and rdata[t_steadystate] != np.nan:
        logger.debug(f"t_steadystate: {rdata[t_steadystate]}")

    logger.debug(f"res: {rdata['res']}")


def map_par_opt_to_par_sim(
        condition_map_sim_var: Dict[str, Union[float, str]],
        x_dct: Dict[str, float],
        amici_model: AmiciModel
) -> np.ndarray:
    """
    From the optimization vector, create the simulation vector according
    to the mapping.

    Parameters
    ----------

    condition_map_sim_var:
        Simulation to optimization parameter mapping.
    x_dct:
        The optimization parameters dict.
    amici_model:
        The amici model.

    Returns
    -------

    par_sim_vals:
        The simulation parameters vector corresponding to x under the
        specified mapping.
    """
    par_sim_vals = [condition_map_sim_var[par_id]
                    for par_id in amici_model.getParameterIds()]

    # iterate over simulation parameter indices
    for ix, val in enumerate(par_sim_vals):
        if not isinstance(val, numbers.Number):
            # value is optimization parameter id
            par_sim_vals[ix] = x_dct[val]

    # return the created simulation parameter vector
    return np.array(par_sim_vals)


def create_plist_from_par_opt_to_par_sim(mapping_par_opt_to_par_sim):
    """
    From the parameter mapping `mapping_par_opt_to_par_sim`, create the
    simulation plist according to the mapping `mapping`.

    Parameters
    ----------

    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.

    Returns
    -------

    plist: array-like of float
        List of parameter indices for which the sensitivity needs to be
        computed
    """
    plist = []

    # iterate over simulation parameter indices
    for j_par_sim, val in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(val, numbers.Number):
            plist.append(j_par_sim)

    # return the created simulation parameter vector
    return plist


def create_identity_parameter_mapping(
        amici_model: AmiciModel, n_conditions: int
) -> 'ParameterMapping':
    """Create a dummy identity parameter mapping table.

    This fills in only the dynamic parameters. Values for fixed parameters,
    both in preequilibration and simulation, are assumed to be provided
    correctly in model or edatas already.
    """
    x_ids = list(amici_model.getParameterIds())
    x_scales = list(amici_model.getParameterScale())
    parameter_mapping = ParameterMapping()
    for _ in range(n_conditions):
        condition_map_sim_var = {x_id: x_id for x_id in x_ids}
        condition_scale_map_sim_var = {
            x_id: amici.parameter_mapping.amici_to_petab_scale(x_scale)
            for x_id, x_scale in zip(x_ids, x_scales)}
        # assumes fixed parameters are filled in already
        mapping_for_condition = ParameterMappingForCondition(
            map_sim_var=condition_map_sim_var,
            scale_map_sim_var=condition_scale_map_sim_var)

        parameter_mapping.append(mapping_for_condition)
    return parameter_mapping


def add_sim_grad_to_opt_grad(
        par_opt_ids: Sequence[str],
        par_sim_ids: Sequence[str],
        condition_map_sim_var: Dict[str, Union[float, str]],
        sim_grad: Sequence[float],
        opt_grad: Sequence[float],
        coefficient: float = 1.0):
    """
    Sum simulation gradients to objective gradient according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    par_opt_ids:
        The optimization parameter ids. Needed for order.
    par_sim_ids:
        The simulation parameter ids. Needed for order.
    condition_map_sim_var:
        The simulation to optimization parameter mapping.
    sim_grad:
        Simulation gradient.
    opt_grad:
        The optimization gradient. To which sim_grad is added.
        Changed in-place.
    coefficient:
        Coefficient for sim_grad when adding to opt_grad.
    """
    for par_sim, par_opt in condition_map_sim_var.items():
        if not isinstance(par_opt, str):
            continue
        par_sim_idx = par_sim_ids.index(par_sim)
        par_opt_idx = par_opt_ids.index(par_opt)

        opt_grad[par_opt_idx] += coefficient * sim_grad[par_sim_idx]


def add_sim_hess_to_opt_hess(
        par_opt_ids: Sequence[str],
        par_sim_ids: Sequence[str],
        condition_map_sim_var: Dict[str, Union[float, str]],
        sim_hess: np.ndarray,
        opt_hess: np.ndarray,
        coefficient: float = 1.0):
    """
    Sum simulation hessians to objective hessian according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    Same as for add_sim_grad_to_opt_grad, replacing the gradients by hessians.
    """
    for par_sim_id, par_opt_id in condition_map_sim_var.items():
        if not isinstance(par_opt_id, str):
            continue
        par_sim_idx = par_sim_ids.index(par_sim_id)
        par_opt_idx = par_opt_ids.index(par_opt_id)

        for par_sim_id_2, par_opt_id_2 in condition_map_sim_var.items():
            if not isinstance(par_opt_id_2, str):
                continue
            par_sim_idx_2 = par_sim_ids.index(par_sim_id_2)
            par_opt_idx_2 = par_opt_ids.index(par_opt_id_2)

            opt_hess[par_opt_idx, par_opt_idx_2] += \
                coefficient * sim_hess[par_sim_idx, par_sim_idx_2]


def sim_sres_to_opt_sres(par_opt_ids: Sequence[str],
                         par_sim_ids: Sequence[str],
                         condition_map_sim_var: Dict[str, Union[float, str]],
                         sim_sres: np.ndarray,
                         coefficient: float = 1.0):
    """
    Sum simulation residual sensitivities to objective residual sensitivities
    according to the provided mapping.

    Parameters
    ----------

    Mostly the same as for add_sim_grad_to_opt_grad, replacing the gradients by
    residual sensitivities.
    """
    opt_sres = np.zeros((sim_sres.shape[0], len(par_opt_ids)))

    for par_sim_id, par_opt_id in condition_map_sim_var.items():
        if not isinstance(par_opt_id, str):
            continue

        par_sim_idx = par_sim_ids.index(par_sim_id)
        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_sres[:, par_opt_idx] += \
            coefficient * sim_sres[:, par_sim_idx]

    return opt_sres

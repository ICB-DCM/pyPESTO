import numpy as np
import copy
import logging
import numbers
from .objective import Objective
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS

try:
    import amici
except ImportError:
    amici = None

logger = logging.getLogger(__name__)


class AmiciObjective(Objective):
    """
    This class allows to create an objective directly from an amici model.
    """

    def __init__(self,
                 amici_model, amici_solver, edatas,
                 max_sensi_order=None,
                 x_ids=None, x_names=None,
                 mapping_par_opt_to_par_sim=None,
                 mapping_scale_opt_to_scale_sim=None,
                 guess_steadystate=True,
                 n_threads=1,
                 options=None):
        """
        Constructor.

        Parameters
        ----------

        amici_model: amici.Model
            The amici model.

        amici_solver: amici.Solver
            The solver to use for the numeric integration of the model.

        edatas: amici.ExpData or list of amici.ExpData
            The experimental data. If a list is passed, its entries correspond
            to multiple experimental conditions.

        max_sensi_order: int, optional
            Maximum sensitivity order supported by the model. Defaults to 2 if
            the model was compiled with o2mode, otherwise 1.

        x_ids: list of str, optional
            Ids of optimization parameters. In the simplest case, this will be
            the AMICI model parameters (default).

        x_names: list of str, optional
            See ``Objective.x_names``.

        mapping_par_opt_to_par_sim: optional
            Mapping of optimization parameters to model parameters. List array
            of size n_simulation_parameters * n_conditions.
            The default is just to assume that optimization and simulation
            parameters coincide. The default is to assume equality of both.

        mapping_scale_opt_to_scale_sim: optional
            Mapping of optimization parameter scales to simulation parameter
            scales. The default is to just use the scales specified in the
            `amici_model` already.

        guess_steadystate: bool, optional (default = True)
            Whether to guess steadystates based on previous steadystates and
            respective derivatives. This option may lead to unexpected
            results for models with conservation laws and should accordingly
            be deactivated for those models.

        n_threads: int, optional (default = 1)
            Number of threads that are used for parallelization over
            experimental conditions. If amici was not installed with openMP
            support this option will have no effect.

        options: pypesto.ObjectiveOptions, optional
            Further options.
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
            options=options
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
        if mapping_par_opt_to_par_sim is None:
            # use identity mapping for each condition
            mapping_par_opt_to_par_sim = \
                [x_ids for _ in range(len(self.edatas))]
        self.mapping_par_opt_to_par_sim = mapping_par_opt_to_par_sim

        # mapping of parameter scales
        if mapping_scale_opt_to_scale_sim is None:
            # use scales from amici model
            mapping_scale_opt_to_scale_sim = \
                create_scale_mapping_from_model(
                    self.amici_model.getParameterScale(), len(self.edatas))
        self.mapping_scale_opt_to_scale_sim = mapping_scale_opt_to_scale_sim

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
                    self.amici_solver.getNewtonPreequilibration()
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

    def __deepcopy__(self, memodict=None):
        model = amici.ModelPtr(self.amici_model.clone())
        solver = amici.SolverPtr(self.amici_solver.clone())
        edatas = [amici.ExpData(data) for data in self.edatas]
        other = AmiciObjective(model, solver, edatas,
                               guess_steadystate=self.guess_steadystate)
        for attr in self.__dict__:
            if attr not in ['amici_solver', 'amici_model', 'edatas']:
                other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    def reset(self):
        """
        Resets the objective, including steadystate guesses
        """
        super(AmiciObjective, self).reset()
        self.reset_steadystate_guesses()

    def _call_amici(
            self,
            x,
            sensi_orders,
            mode
    ):
        # amici is built such that only the maximum sensitivity is required,
        # the lower orders are then automatically computed
        sensi_order = min(max(sensi_orders), 1)
        # order 2 currently not implemented, we are using the FIM

        # check if the requested sensitivities can be computed
        if sensi_order > self.max_sensi_order:
            raise Exception("Sensitivity order not allowed.")

        # prepare outputs
        nllh = 0.0
        snllh = np.zeros(self.dim)
        s2nllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        # set order in solver
        self.amici_solver.setSensitivityOrder(sensi_order)

        # loop over experimental data
        for data_ix, edata in enumerate(self.edatas):

            # set model parameter scale for condition index
            self.set_parameter_scale(data_ix)

            # set parameters in model, according to mapping
            self.set_par_sim_for_condition(data_ix, x)

            # set parameter list according to mapping
            self.set_plist_for_condition(data_ix)

            if self.guess_steadystate and \
                    self.steadystate_guesses['fval'] < np.inf:
                self.apply_steadystate_guess(data_ix, x)

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            self.amici_model,
            self.amici_solver,
            self.edatas,
            num_threads=min(self.n_threads, len(self.edatas)),
        )

        for data_ix, rdata in enumerate(rdatas):
            log_simulation(data_ix, rdata)

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(rdatas)

            # compute objective
            if mode == MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    add_sim_grad_to_opt_grad(
                        self.x_ids,
                        self.mapping_par_opt_to_par_sim[data_ix],
                        rdata['sllh'],
                        snllh,
                        coefficient=-1.0
                    )
                    # TODO: Compute the full Hessian, and check here
                    add_sim_hess_to_opt_hess(
                        self.x_ids,
                        self.mapping_par_opt_to_par_sim[data_ix],
                        rdata['FIM'],
                        s2nllh,
                        coefficient=-1.0
                    )

            elif mode == MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    opt_sres = sim_sres_to_opt_sres(
                        self.x_ids,
                        self.mapping_par_opt_to_par_sim[data_ix],
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
                self.store_steadystate_guess(data_ix, x, rdata)

        return {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }

    def get_error_output(self, rdatas):
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

    def set_par_sim_for_condition(self, condition_ix, x):
        """
        Set the simulation parameters from the optimization parameters
        for the given condition.

        Parameters
        ----------

        condition_ix: int
            Index of the current experimental condition.

        x: array_like
            Optimization parameters.
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
        self.edatas[condition_ix].parameters = x_sim

    def set_plist_for_condition(self, condition_ix):
        """
        Set the plist according to the optimization parameters
        for the given condition.

        Parameters
        ----------

        condition_ix: int
            Index of the current experimental condition.

        x: array_like
            Optimization parameters.
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        plist = create_plist_from_par_opt_to_par_sim(mapping)
        self.edatas[condition_ix].plist = plist

    def set_parameter_scale(self, condition_ix):
        scale_list = self.mapping_scale_opt_to_scale_sim[condition_ix]
        amici_scale_vector = amici.ParameterScalingVector()

        for val in scale_list:

            if val == 'lin':
                scale = amici.ParameterScaling_none
            elif val == 'log10':
                scale = amici.ParameterScaling_log10
            elif val == 'log':
                scale = amici.ParameterScaling_ln
            else:
                raise ValueError(
                    f"Parameter scaling not recognized: {val}")

            # append to scale vector
            amici_scale_vector.append(scale)

        self.edatas[condition_ix].pscale = amici_scale_vector

    def apply_steadystate_guess(self, condition_ix, x):
        """
        Use the stored steadystate as well as the respective  sensitivity (
        if available) and parameter value to approximate the steadystate at
        the current parameters using a zeroth or first order taylor
        approximation:
        x_ss(x') = x_ss(x) [+ dx_ss/dx(x)*(x'-x)]
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
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

    def store_steadystate_guess(self, condition_ix, x, rdata):
        """
        Store condition parameter, steadystate and steadystate sensitivity in
        steadystate_guesses if steadystate guesses are enabled for this
        condition
        """

        if condition_ix not in self.steadystate_guesses['data']:
            return

        preeq_guesses = self.steadystate_guesses['data'][condition_ix]

        # update parameter

        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
        preeq_guesses['x'] = x_sim

        # update steadystates
        preeq_guesses['x_ss'] = rdata['x_ss']
        preeq_guesses['sx_ss'] = rdata['sx_ss']

    def reset_steadystate_guesses(self):
        """
        Resets all steadystate guess data
        """
        if not self.guess_steadystate:
            return

        self.steadystate_guesses['fval'] = np.inf
        for condition in self.steadystate_guesses['data']:
            self.steadystate_guesses['data'][condition] = dict()


def log_simulation(data_ix, rdata):
    """
    Log the simulation results.
    """
    logger.debug(f"=== DATASET {data_ix} ===")
    logger.debug(f"status: {rdata['status']}")
    logger.debug(f"llh: {rdata['llh']}")

    t_steadystate = 't_steadystate'
    if t_steadystate in rdata and rdata[t_steadystate] != np.nan:
        logger.debug(f"t_steadystate: {rdata[t_steadystate]}")

    logger.debug(f"res: {rdata['res']}")


def map_par_opt_to_par_sim(mapping_par_opt_to_par_sim, par_opt_ids, x):
    """
    From the optimization vector `x`, create the simulation vector according
    to the mapping `mapping`.

    Parameters
    ----------

    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.
    par_opt_ids: array-like of str
        The optimization parameter ids. This vector is needed to know the
        order of the entries in x.
    x: array-like of float
        The optimization parameters vector.

    Returns
    -------

    y: array-like of float
        The simulation parameters vector corresponding to x under the
        specified mapping.
    """

    # number of simulation parameters
    n_par_sim = len(mapping_par_opt_to_par_sim)

    # prepare simulation parameter vector
    par_sim_vals = np.zeros(n_par_sim)

    # iterate over simulation parameter indices
    for j_par_sim in range(n_par_sim):
        # extract entry in mapping table for j_par_sim
        val = mapping_par_opt_to_par_sim[j_par_sim]

        if isinstance(val, numbers.Number):
            # fixed value assignment
            par_sim_vals[j_par_sim] = val
        else:
            # value is optimization parameter id
            par_sim_vals[j_par_sim] = x[par_opt_ids.index(val)]

    # return the created simulation parameter vector
    return par_sim_vals


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


def create_scale_mapping_from_model(amici_scales, n_edata):
    """
    Create parameter scaling mapping matrix from amici scaling
    vector.
    """
    scales = []
    amici_scales = list(amici_scales)

    for amici_scale in amici_scales:
        if amici_scale == amici.ParameterScaling_none:
            scale = 'lin'
        elif amici_scale == amici.ParameterScaling_ln:
            scale = 'log'
        elif amici_scale == amici.ParameterScaling_log10:
            scale = 'log10'
        else:
            raise Exception(
                f"Parameter scaling {amici_scale} in amici model not"
                f"recognized.")
        scales.append(scale)

    mapping_scale_opt_to_scale_sim = [scales for _ in range(n_edata)]

    return mapping_scale_opt_to_scale_sim


def add_sim_grad_to_opt_grad(par_opt_ids,
                             mapping_par_opt_to_par_sim,
                             sim_grad,
                             opt_grad,
                             coefficient: float = 1.0):
    """
    Sum simulation gradients to objective gradient according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    par_opt_ids: array-like of str
        The optimization parameter ids. This vector is needed to know the
        order of the entries in x.
    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.
    sim_grad: array-like of float
        Simulation gradient.
    opt_grad: array-like of float
        The optimization gradient. To which sim_grad is added.
        Will be changed in place.
    coefficient: float
        Coefficient for sim_grad when adding to opt_grad.
    """

    par_sim_idx = 0
    for par_opt_id in mapping_par_opt_to_par_sim:
        # we ignore non-string indices as a fixed value as been set for
        # those and they are not included in the condition specific nplist,
        # we do not only skip here, but also do not increase par_sim_idx!
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the gradient
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_grad[par_opt_idx] += coefficient * sim_grad[par_sim_idx]
        par_sim_idx += 1


def add_sim_hess_to_opt_hess(par_opt_ids,
                             mapping_par_opt_to_par_sim,
                             sim_hess,
                             opt_hess,
                             coefficient: float = 1.0):
    """
    Sum simulation hessians to objective hessian according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    Same as for add_sim_grad_to_opt_grad, replacing the gradients by hessians.
    """

    # use enumerate for first axis as plist is not applied
    # https://github.com/ICB-DCM/AMICI/issues/274
    par_sim_idx = 0
    for par_opt_id in mapping_par_opt_to_par_sim:
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the hessian
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)

        # for second axis, plist was applied so we can skip over values with
        # numeric mapping
        par_sim_idx_2 = 0
        for par_opt_id_2 in mapping_par_opt_to_par_sim:
            # we ignore non-string indices as a fixed value as been set for
            # those and they are not included in the condition specific nplist,
            # we not only skip here, but also do not increase par_sim_idx_2!
            if not isinstance(par_opt_id_2, str):
                continue

            par_opt_idx_2 = par_opt_ids.index(par_opt_id_2)

            opt_hess[par_opt_idx, par_opt_idx_2] += \
                coefficient * sim_hess[par_sim_idx, par_sim_idx_2]
            par_sim_idx_2 += 1
        par_sim_idx += 1


def sim_sres_to_opt_sres(par_opt_ids,
                         mapping_par_opt_to_par_sim,
                         sim_sres,
                         coefficient: float = 1.0):
    """
    Sum simulation residual sensitivities to objective residual sensitivities
    according to the provided mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    Same as for add_sim_grad_to_opt_grad, replacing the gradients by residual
    sensitivities.
    """
    opt_sres = np.zeros((sim_sres.shape[0], len(par_opt_ids)))

    par_sim_idx = 0
    for par_opt_id in mapping_par_opt_to_par_sim:
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the hessian
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_sres[:, par_opt_idx] += \
            coefficient * sim_sres[:, par_sim_idx]
        par_sim_idx += 1

    return opt_sres

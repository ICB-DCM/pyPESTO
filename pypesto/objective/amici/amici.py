import abc
import copy
import os
import tempfile
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ...C import FVAL, INNER_PARAMETERS, MODE_FUN, MODE_RES, RDATAS, ModeType
from ..base import ObjectiveBase, ResultDict
from .amici_calculator import AmiciCalculator
from .amici_util import (
    create_identity_parameter_mapping,
    map_par_opt_to_par_sim,
)

if TYPE_CHECKING:
    try:
        import amici
        from amici.parameter_mapping import ParameterMapping
    except ImportError:
        pass

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


class AmiciObjectBuilder(abc.ABC):
    """Allows to build AMICI model, solver, and edatas.

    This class is useful for pickling an :class:`pypesto.AmiciObjective`,
    which is required in some parallelization schemes. Therefore, this
    class itself must be picklable.
    """

    @abc.abstractmethod
    def create_model(self) -> AmiciModel:
        """Create an AMICI model."""

    @abc.abstractmethod
    def create_solver(self, model: AmiciModel) -> AmiciSolver:
        """Create an AMICI solver."""

    @abc.abstractmethod
    def create_edatas(self, model: AmiciModel) -> Sequence['amici.ExpData']:
        """Create AMICI experimental data."""


class AmiciObjective(ObjectiveBase):
    """Allows to create an objective directly from an amici model."""

    def __init__(
        self,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: Union[Sequence['amici.ExpData'], 'amici.ExpData'],
        max_sensi_order: Optional[int] = None,
        x_ids: Optional[Sequence[str]] = None,
        x_names: Optional[Sequence[str]] = None,
        parameter_mapping: Optional['ParameterMapping'] = None,
        guess_steadystate: Optional[Optional[bool]] = None,
        n_threads: Optional[int] = 1,
        fim_for_hess: Optional[bool] = True,
        amici_object_builder: Optional[AmiciObjectBuilder] = None,
        calculator: Optional[AmiciCalculator] = None,
        amici_reporting: Optional['amici.RDataReporting'] = None,
    ):
        """
        Initialize objective.

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
        fim_for_hess:
            Whether to use the FIM whenever the Hessian is requested. This only
            applies with forward sensitivities.
            With adjoint sensitivities, the true Hessian will be used,
            if available.
            FIM or Hessian will only be exposed if `max_sensi_order>1`.
        amici_object_builder:
            AMICI object builder. Allows recreating the objective for
            pickling, required in some parallelization schemes.
        calculator:
            Performs the actual calculation of the function values and
            derivatives.
        amici_reporting:
            Determines which quantities will be computed by AMICI,
            see ``amici.Solver.setReturnDataReportingMode``. Set to ``None``
            to compute only the minimum required information.
        """
        import amici

        if amici is None:
            raise ImportError(
                "This objective requires an installation of amici "
                "(https://github.com/icb-dcm/amici). "
                "Install via `pip3 install amici`."
            )

        self.amici_model = amici_model.clone()
        self.amici_solver = amici_solver.clone()

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

        # mapping of parameters
        if parameter_mapping is None:
            # use identity mapping for each condition
            parameter_mapping = create_identity_parameter_mapping(
                amici_model, len(edatas)
            )
        self.parameter_mapping = parameter_mapping

        # If supported, enable `guess_steadystate` by default. If not
        #  supported, disable by default. If requested but unsupported, raise.
        if (
            self.guess_steadystate is not False
            and self.amici_model.nx_solver_reinit > 0
        ):
            if self.guess_steadystate:
                raise ValueError(
                    'Steadystate prediction is not supported '
                    'for models with conservation laws!'
                )
            self.guess_steadystate = False

        if (
            self.guess_steadystate is not False
            and self.amici_model.getSteadyStateSensitivityMode()
            == amici.SteadyStateSensitivityMode.integrationOnly
        ):
            if self.guess_steadystate:
                raise ValueError(
                    'Steadystate guesses cannot be enabled '
                    'when `integrationOnly` as '
                    'SteadyStateSensitivityMode!'
                )
            self.guess_steadystate = False

        if self.guess_steadystate is not False:
            self.guess_steadystate = True

        if self.guess_steadystate:
            # preallocate guesses, construct a dict for every edata for which
            #  we need to do preequilibration
            self.steadystate_guesses = {
                'fval': np.inf,
                'data': {
                    iexp: {}
                    for iexp, edata in enumerate(self.edatas)
                    if len(edata.fixedParametersPreequilibration)
                },
            }
        # optimization parameter names
        if x_names is None:
            # use ids as names
            x_names = x_ids

        self.n_threads = n_threads
        self.fim_for_hess = fim_for_hess
        self.amici_object_builder = amici_object_builder
        self.amici_reporting = amici_reporting

        if calculator is None:
            calculator = AmiciCalculator()
        self.calculator = calculator
        super().__init__(x_names=x_names)

        # Custom (condition-specific) timepoints. See the
        # `set_custom_timepoints` method for more information.
        self.custom_timepoints = None

        # Initialize the dictionary for saving of inner parameters.
        self.inner_parameters: Dict[str, float] = {}

    def get_config(self) -> dict:
        """Return basic information of the objective configuration."""
        info = super().get_config()
        info['x_names'] = self.x_names
        info['model_name'] = self.amici_model.getName()
        info['solver'] = str(type(self.amici_solver))
        info['sensi_order'] = self.max_sensi_order

        return info

    def initialize(self):
        """See `ObjectiveBase` documentation."""
        super().initialize()
        self.reset_steadystate_guesses()
        self.calculator.initialize()

    def __deepcopy__(self, memodict: Dict = None) -> 'AmiciObjective':
        import amici

        other = self.__class__.__new__(self.__class__)

        for key in set(self.__dict__.keys()) - {
            'amici_model',
            'amici_solver',
            'edatas',
        }:
            other.__dict__[key] = copy.deepcopy(self.__dict__[key])

        # copy objects that do not have __deepcopy__
        other.amici_model = self.amici_model.clone()
        other.amici_solver = self.amici_solver.clone()
        other.edatas = [amici.ExpData(data) for data in self.edatas]

        return other

    def __getstate__(self) -> Dict:
        import amici

        if self.amici_object_builder is None:
            raise NotImplementedError(
                "AmiciObjective does not support __getstate__ without "
                "an `amici_object_builder`."
            )

        state = {}
        for key in set(self.__dict__.keys()) - {
            'amici_model',
            'amici_solver',
            'edatas',
        }:
            state[key] = self.__dict__[key]

        _fd, _file = tempfile.mkstemp()
        try:
            # write amici solver settings to file
            try:
                amici.writeSolverSettingsToHDF5(self.amici_solver, _file)
            except AttributeError as e:
                e.args += (
                    "Pickling the AmiciObjective requires an AMICI "
                    "installation with HDF5 support.",
                )
                raise
            # read in byte stream
            with open(_fd, 'rb', closefd=False) as f:
                state['amici_solver_settings'] = f.read()
        finally:
            # close file descriptor and remove temporary file
            os.close(_fd)
            os.remove(_file)

        state['AMICI_model_settings'] = amici.get_model_settings(
            self.amici_model
        )

        return state

    def __setstate__(self, state: Dict) -> None:
        import amici

        if state['amici_object_builder'] is None:
            raise NotImplementedError(
                "AmiciObjective does not support __setstate__ without "
                "an `amici_object_builder`."
            )
        self.__dict__.update(state)

        # note: attributes not defined in the builder are lost
        model = self.amici_object_builder.create_model()
        solver = self.amici_object_builder.create_solver(model)
        edatas = self.amici_object_builder.create_edatas(model)

        _fd, _file = tempfile.mkstemp()
        try:
            # write solver settings to temporary file
            with open(_fd, 'wb', closefd=False) as f:
                f.write(state['amici_solver_settings'])
            # read in solver settings
            try:
                amici.readSolverSettingsFromHDF5(_file, solver)
            except AttributeError as err:
                if not err.args:
                    err.args = ('',)
                err.args += (
                    "Unpickling an AmiciObjective requires an AMICI "
                    "installation with HDF5 support.",
                )
                raise
        finally:
            # close file descriptor and remove temporary file
            os.close(_fd)
            os.remove(_file)

        self.amici_model = model
        self.amici_solver = solver
        self.edatas = edatas

        self.apply_custom_timepoints()
        amici.set_model_settings(
            self.amici_model,
            state['AMICI_model_settings'],
        )

    def check_sensi_orders(
        self,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
    ) -> bool:
        """See `ObjectiveBase` documentation."""
        import amici

        if not sensi_orders:
            return True
        sensi_order = max(sensi_orders)

        # dynamically obtain maximum allowed sensitivity order
        max_sensi_order = self.max_sensi_order
        if max_sensi_order is None:
            max_sensi_order = 1
            # check whether it is ok to request 2nd order
            sensi_mthd = self.amici_solver.getSensitivityMethod()
            mthd_fwd = amici.SensitivityMethod_forward
            if mode == MODE_FUN and (
                self.amici_model.o2mode
                or (sensi_mthd == mthd_fwd and self.fim_for_hess)
            ):
                max_sensi_order = 2

        # evaluate sensitivity order
        return sensi_order <= max_sensi_order

    def check_mode(self, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        return mode in [MODE_FUN, MODE_RES]

    def __call__(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...] = (0,),
        mode: ModeType = MODE_FUN,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray, Tuple, ResultDict]:
        """See `ObjectiveBase` documentation."""
        import amici

        # Use AMICI full reporting if amici.ReturnDatas are returned and no
        #  other reporting mode was set
        if (
            return_dict
            and self.amici_reporting is None
            and 'amici_reporting' not in kwargs
        ):
            kwargs['amici_reporting'] = amici.RDataReporting.full

        return super().__call__(x, sensi_orders, mode, return_dict, **kwargs)

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        edatas: Sequence['amici.ExpData'] = None,
        parameter_mapping: 'ParameterMapping' = None,
        amici_reporting: Optional['amici.RDataReporting'] = None,
    ):
        """
        Call objective function without pre- or post-processing and formatting.

        Returns
        -------
        result:
            A dict containing the results.
        """
        import amici

        x_dct = self.par_arr_to_dct(x)

        # only ask amici to compute required quantities
        amici_reporting = (
            self.amici_reporting
            if amici_reporting is None
            else amici_reporting
        )
        if amici_reporting is None:
            amici_reporting = (
                amici.RDataReporting.likelihood
                if mode == MODE_FUN
                else amici.RDataReporting.residuals
            )
        self.amici_solver.setReturnDataReportingMode(amici_reporting)

        # update steady state
        if (
            self.guess_steadystate
            and self.steadystate_guesses['fval'] < np.inf
        ):
            for data_ix in range(len(self.edatas)):
                self.apply_steadystate_guess(data_ix, x_dct)

        if edatas is None:
            edatas = self.edatas
        if parameter_mapping is None:
            parameter_mapping = self.parameter_mapping
        ret = self.calculator(
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=self.amici_model,
            amici_solver=self.amici_solver,
            edatas=edatas,
            n_threads=self.n_threads,
            x_ids=self.x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=self.fim_for_hess,
        )

        nllh = ret[FVAL]
        rdatas = ret[RDATAS]
        if INNER_PARAMETERS in ret and ret[INNER_PARAMETERS]:
            self.inner_parameters = ret[INNER_PARAMETERS]

        # check whether we should update data for preequilibration guesses
        if (
            self.guess_steadystate
            and nllh <= self.steadystate_guesses['fval']
            and nllh < np.inf
        ):
            self.steadystate_guesses['fval'] = nllh
            for data_ix, rdata in enumerate(rdatas):
                self.store_steadystate_guess(data_ix, x_dct, rdata)

        return ret

    def par_arr_to_dct(self, x: Sequence[float]) -> Dict[str, float]:
        """Create dict from parameter vector."""
        return OrderedDict(zip(self.x_ids, x))

    def apply_steadystate_guess(self, condition_ix: int, x_dct: Dict) -> None:
        """
        Apply steady state guess to `edatas[condition_ix].x0`.

        Use the stored steadystate as well as the respective sensitivity (
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
                linear_update = (
                    guess_data['sx_ss']
                    .transpose()
                    .dot(
                        (x_sim - guess_data['x'])[
                            np.asarray(self.edatas[condition_ix].plist)
                        ]
                    )
                )
                # limit linear updates to max 20 % elementwise change
                if (linear_update / (x_ss_guess + np.spacing(1))).max() < 0.2:
                    x_ss_guess += linear_update

        self.edatas[condition_ix].x0 = tuple(x_ss_guess)

    def store_steadystate_guess(
        self,
        condition_ix: int,
        x_dct: Dict,
        rdata: 'amici.ReturnData',
    ) -> None:
        """
        Store condition parameter, steadystate and steadystate sensitivity.

        Stored in steadystate_guesses if steadystate guesses are enabled for
        this condition.
        """
        if condition_ix not in self.steadystate_guesses['data']:
            return
        preeq_guesses = self.steadystate_guesses['data'][condition_ix]

        # update parameter
        condition_map_sim_var = self.parameter_mapping[
            condition_ix
        ].map_sim_var
        x_sim = map_par_opt_to_par_sim(
            condition_map_sim_var, x_dct, self.amici_model
        )
        preeq_guesses['x'] = x_sim

        # update steadystates
        preeq_guesses['x_ss'] = rdata['x_ss']
        preeq_guesses['sx_ss'] = rdata['sx_ss']

    def reset_steadystate_guesses(self) -> None:
        """Reset all steadystate guess data."""
        if not self.guess_steadystate:
            return

        self.steadystate_guesses['fval'] = np.inf
        for condition in self.steadystate_guesses['data']:
            self.steadystate_guesses['data'][condition] = {}

    def apply_custom_timepoints(self) -> None:
        """Apply custom timepoints, if applicable.

        See the `set_custom_timepoints` method for more information.
        """
        if self.custom_timepoints is not None:
            for index in range(len(self.edatas)):
                self.edatas[index].setTimepoints(self.custom_timepoints[index])

    def set_custom_timepoints(
        self,
        timepoints: Sequence[Sequence[Union[float, int]]] = None,
        timepoints_global: Sequence[Union[float, int]] = None,
    ) -> 'AmiciObjective':
        """
        Create a copy of this objective that is evaluated at custom timepoints.

        The intended use is to aid in predictions at unmeasured timepoints.

        Parameters
        ----------
        timepoints:
            The outer sequence should contain a sequence of timepoints for each
            experimental condition.
        timepoints_global:
            A sequence of timepoints that will be used for all experimental
            conditions.

        Returns
        -------
        The customized copy of this objective.
        """
        if timepoints is None and timepoints_global is None:
            raise KeyError('Timepoints were not specified.')

        amici_objective = copy.deepcopy(self)

        if timepoints is not None:
            if len(timepoints) != len(amici_objective.edatas):
                raise ValueError(
                    'The number of condition-specific timepoints `timepoints` '
                    'does not match the number of experimental conditions.\n'
                    f'Number of provided timepoints: {len(timepoints)}. '
                    'Number of experimental conditions: '
                    f'{len(amici_objective.edatas)}.'
                )
            custom_timepoints = timepoints
        else:
            custom_timepoints = [
                copy.deepcopy(timepoints_global)
                for _ in range(len(amici_objective.edatas))
            ]

        amici_objective.custom_timepoints = custom_timepoints
        amici_objective.apply_custom_timepoints()
        return amici_objective

    def check_gradients_match_finite_differences(
        self, x: np.ndarray = None, *args, **kwargs
    ) -> bool:
        """Check if gradients match finite differences (FDs).

        Parameters
        ----------
        x: The parameters for which to evaluate the gradient.

        Returns
        -------
        bool
            Indicates whether gradients match (True) FDs or not (False)
        """
        if x is None and 'petab_problem' in dir(self.amici_object_builder):
            x = self.amici_object_builder.petab_problem.x_nominal_scaled
            x_free = self.amici_object_builder.petab_problem.x_free_indices
        return super().check_gradients_match_finite_differences(
            x=x, x_free=x_free, *args, **kwargs
        )

"""Module for the InnerCalculatorCollector class.

In case of semi-quantitative or qualitative measurements, this class is used
to collect hierarchical inner calculators for each data type and merge their results.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import amici.sim.sundials.petab
    from petab import v2

from ..C import (
    AMICI_SIGMAY,
    AMICI_SSIGMAY,
    AMICI_SY,
    AMICI_Y,
    CENSORED,
    FVAL,
    GRAD,
    HESS,
    INNER_PARAMETERS,
    LAPLACE,
    LIN,
    LOG,
    LOG10,
    METHOD,
    MODE_RES,
    NORMAL,
    ORDINAL,
    ORDINAL_OPTIONS,
    RDATAS,
    RELATIVE,
    RES,
    SEMIQUANTITATIVE,
    SPLINE_APPROXIMATION_OPTIONS,
    SPLINE_KNOTS,
    SPLINE_RATIO,
    SRES,
    ModeType,
)
from ..objective.amici.amici_calculator import AmiciCalculator
from ..objective.amici.amici_util import (
    filter_return_dict,
    init_return_values,
    par_index_slices,
    petab_v2_index_slices,
)

try:
    import amici.sim.sundials as asd
    import petab.v1 as petab
    from amici.sim._parameter_mapping import ParameterMapping
except ImportError:
    petab = None
    ParameterMapping = None

from .ordinal import OrdinalCalculator, OrdinalInnerSolver, OrdinalProblem
from .relative import RelativeAmiciCalculator, RelativeInnerProblem
from .semiquantitative import (
    SemiquantCalculator,
    SemiquantInnerSolver,
    SemiquantProblem,
)

AmiciModel = Union["asd.Model", "asd.ModelPtr"]
AmiciSolver = Union["asd.Solver", "asd.SolverPtr"]


class InnerCalculatorCollector(AmiciCalculator):
    """Class to collect inner calculators in case of non-quantitative data types.

    Upon import of a petab problem, the PEtab importer checks whether there are
    non-quantitative data types. If so, it creates an instance of this class
    instead of an AmiciCalculator. This class then collects the inner calculators
    for each data type and merges their results with the quantitative results.

    Parameters
    ----------
    data_types:
        List of non-quantitative data types in the problem.
    petab_problem:
        The PEtab problem.
    model:
        The AMICI model.
    edatas:
        The experimental data.
    inner_options:
        Options for the inner problems and solvers.
    """

    def __init__(
        self,
        data_types: set[str],
        petab_problem: petab.Problem,
        model: AmiciModel,
        edatas: list[asd.ExpData],
        inner_options: dict,
    ):
        super().__init__()
        self.validate_options(inner_options)

        self.data_types = data_types
        self.inner_calculators: list[
            AmiciCalculator
        ] = []  # TODO make into a dictionary (future PR, together with .hierarchical of Problem)

        self.semiquant_observable_ids = None
        self.relative_observable_ids = None

        self.construct_inner_calculators(
            petab_problem, model, edatas, inner_options
        )

        self.quantitative_data_mask = self._get_quantitative_data_mask(edatas)
        #: per model observable, its ``(transformation, distribution)``
        self.noise_models = self._get_noise_models(petab_problem, model)

        #: per condition, the ``(par_sim_slice, par_opt_slice)`` pairs
        #: mapping the sensitivities onto the optimization parameters.
        #: ``None`` means they are derived from the parameter mapping.
        self._index_slices = None

    def initialize(self):
        """Initialize."""
        for calculator in self.inner_calculators:
            calculator.initialize()

    def construct_inner_calculators(
        self,
        petab_problem: petab.Problem,
        model: AmiciModel,
        edatas: list[asd.ExpData],
        inner_options: dict,
    ):
        """Construct inner calculators for each data type."""
        self.necessary_par_dummy_values = {}

        if RELATIVE in self.data_types:
            relative_inner_problem = RelativeInnerProblem.from_petab_amici(
                petab_problem, model, edatas
            )
            self.necessary_par_dummy_values.update(
                relative_inner_problem.get_dummy_values(scaled=True)
            )
            relative_inner_solver = RelativeAmiciCalculator(
                inner_problem=relative_inner_problem
            )
            self.inner_calculators.append(relative_inner_solver)
            self.relative_observable_ids = (
                relative_inner_problem.get_relative_observable_ids()
            )

        if ORDINAL in self.data_types or CENSORED in self.data_types:
            optimal_scaling_inner_options = {
                key: value
                for key, value in inner_options.items()
                if key in ORDINAL_OPTIONS
            }
            inner_problem_method = optimal_scaling_inner_options.get(
                METHOD, None
            )
            ordinal_inner_problem = OrdinalProblem.from_petab_amici(
                petab_problem, model, edatas, inner_problem_method
            )
            ordinal_inner_solver = OrdinalInnerSolver(
                options=optimal_scaling_inner_options
            )
            ordinal_calculator = OrdinalCalculator(
                ordinal_inner_problem, ordinal_inner_solver
            )
            self.inner_calculators.append(ordinal_calculator)

        if SEMIQUANTITATIVE in self.data_types:
            spline_inner_options = {
                key: value
                for key, value in inner_options.items()
                if key in SPLINE_APPROXIMATION_OPTIONS
            }
            spline_ratio = spline_inner_options.pop(SPLINE_RATIO, None)
            semiquant_problem = SemiquantProblem.from_petab_amici(
                petab_problem, model, edatas, spline_ratio
            )
            semiquant_inner_solver = SemiquantInnerSolver(
                options=spline_inner_options
            )
            semiquant_calculator = SemiquantCalculator(
                semiquant_problem, semiquant_inner_solver
            )
            self.necessary_par_dummy_values.update(
                semiquant_problem.get_noise_dummy_values(scaled=True)
            )
            self.inner_calculators.append(semiquant_calculator)
            self.semiquant_observable_ids = (
                semiquant_problem.get_semiquant_observable_ids()
            )

        if self.data_types - {
            RELATIVE,
            ORDINAL,
            CENSORED,
            SEMIQUANTITATIVE,
        }:
            unsupported_data_types = self.data_types - {
                RELATIVE,
                ORDINAL,
                CENSORED,
                SEMIQUANTITATIVE,
            }
            raise NotImplementedError(
                f"Data types {unsupported_data_types} are not supported."
            )

    def validate_options(self, inner_options: dict):
        """Validate the inner options.

        Parameters
        ----------
        inner_options:
            Options for the inner problems and solvers.
        """
        for key in inner_options:
            if (
                key not in ORDINAL_OPTIONS
                and key not in SPLINE_APPROXIMATION_OPTIONS
            ):
                raise ValueError(f"Unknown inner option {key}.")

    def _get_quantitative_data_mask(
        self,
        edatas: list[asd.ExpData],
    ) -> list[np.ndarray] | None:
        """Get the mask of quantitative measurements, one entry per condition.

        Returns ``None`` if the problem has no quantitative data at all, which
        the callers take to mean "no quantitative contribution to add".
        """
        # transform experimental data
        edatas = [asd.ExpDataView(edata)["measurements"] for edata in edatas]

        quantitative_data_mask = [
            np.ones_like(edata, dtype=bool) for edata in edatas
        ]

        # iterate over inner problems
        for calculator in self.inner_calculators:
            inner_parameters = calculator.inner_problem.xs.values()
            # Remove inner parameter masks from quantitative data mask
            for inner_par in inner_parameters:
                for cond_idx, condition_mask in enumerate(
                    quantitative_data_mask
                ):
                    condition_mask[inner_par.ixs[cond_idx]] = False

        # Put to False all entries that have a nan value in the edata
        for condition_mask, edata in zip(
            quantitative_data_mask, edatas, strict=True
        ):
            condition_mask[np.isnan(edata)] = False

        # If there is no quantitative data at all, return None. Individual
        #  conditions without quantitative data are fine -- their (all-False)
        #  mask simply contributes nothing.
        if not any(mask.any() for mask in quantitative_data_mask):
            return None

        return quantitative_data_mask

    @staticmethod
    def _get_noise_models(
        petab_problem: petab.Problem,
        model: AmiciModel,
    ) -> list[tuple[str, str]]:
        """Get the noise model of each model observable.

        Returns the ``(transformation, distribution)`` of each observable, in
        the order of the model observables, for
        :func:`calculate_quantitative_result`.
        """
        noise_models = []
        for observable_id in model.get_observable_ids():
            observable = petab_problem.observable_df.loc[observable_id]
            transformation = observable.get(petab.OBSERVABLE_TRANSFORMATION)
            distribution = observable.get(petab.NOISE_DISTRIBUTION)
            noise_models.append(
                (
                    LIN if petab.is_empty(transformation) else transformation,
                    NORMAL if petab.is_empty(distribution) else distribution,
                )
            )
        return noise_models

    def get_inner_par_ids(self) -> list[str]:
        """Return the ids of inner parameters of all inner problems."""
        return [
            parameter_id
            for inner_calculator in self.inner_calculators
            for parameter_id in inner_calculator.inner_problem.get_x_ids()
        ]

    def get_interpretable_inner_par_ids(self) -> list[str]:
        """Return the ids of interpretable inner parameters of all inner problems.

        See :func:`InnerProblem.get_interpretable_x_ids`.
        """
        return [
            parameter_id
            for inner_calculator in self.inner_calculators
            for parameter_id in inner_calculator.inner_problem.get_interpretable_x_ids()
        ]

    def get_interpretable_inner_par_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the bounds of interpretable inner parameters of all inner problems."""
        lb = []
        ub = []
        for inner_calculator in self.inner_calculators:
            (
                lb_i,
                ub_i,
            ) = inner_calculator.inner_problem.get_interpretable_x_bounds()
            lb.extend(lb_i)
            ub.extend(ub_i)
        return np.asarray(lb), np.asarray(ub)

    def get_interpretable_inner_par_scales(self) -> list[str]:
        """Return the scales of interpretable inner parameters of all inner problems."""
        return [
            scale
            for inner_calculator in self.inner_calculators
            for scale in inner_calculator.inner_problem.get_interpretable_x_scales()
        ]

    def _combine_inner_results(
        self,
        rdatas: list[asd.ReturnDataView],
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
        index_slices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict:
        """Run the inner calculators on ``rdatas`` and assemble the result.

        Shared by the PEtab v1 and v2 collectors: how the simulations are
        produced differs between the versions, what is done with them does
        not.

        Parameters
        ----------
        rdatas:
            The simulation results. The remaining arguments are those of
            :meth:`__call__`, and are forwarded to the inner calculators
            alongside them.
        index_slices:
            Passed on to :func:`calculate_quantitative_result`. ``None``
            derives them from the PEtab v1 parameter mapping.
        """
        dim = len(x_ids)

        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        interpretable_inner_pars = []
        spline_knots = None

        for calculator in self.inner_calculators:
            inner_result = calculator(
                rdatas=rdatas,
                x_dct=x_dct,
                sensi_orders=sensi_orders,
                mode=mode,
                amici_model=amici_model,
                amici_solver=amici_solver,
                edatas=edatas,
                n_threads=n_threads,
                x_ids=x_ids,
                parameter_mapping=parameter_mapping,
                fim_for_hess=fim_for_hess,
            )
            nllh += inner_result[FVAL]
            if 1 in sensi_orders:
                snllh += inner_result[GRAD]
            if (inner_pars := inner_result.get(INNER_PARAMETERS)) is not None:
                interpretable_inner_pars.extend(inner_pars)
            if SPLINE_KNOTS in inner_result:
                spline_knots = inner_result[SPLINE_KNOTS]

        # add the quantitative data contribution
        if self.quantitative_data_mask is not None:
            quantitative_result = calculate_quantitative_result(
                rdatas=rdatas,
                sensi_orders=sensi_orders,
                edatas=edatas,
                mode=mode,
                quantitative_data_mask=self.quantitative_data_mask,
                noise_models=self.noise_models,
                dim=dim,
                parameter_mapping=parameter_mapping,
                par_opt_ids=x_ids,
                par_sim_ids=amici_model.get_free_parameter_ids(),
                index_slices=index_slices,
            )
            nllh += quantitative_result[FVAL]
            if 1 in sensi_orders:
                snllh += quantitative_result[GRAD]

        return filter_return_dict(
            {
                FVAL: nllh,
                GRAD: snllh,
                HESS: s2nllh,
                RES: res,
                SRES: sres,
                RDATAS: rdatas,
                INNER_PARAMETERS: interpretable_inner_pars or None,
                SPLINE_KNOTS: spline_knots,
            }
        )

    def _direct_to_relative_calculator(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
    ) -> dict | None:
        """Delegate to the relative calculator, where it can handle the call.

        For adjoint gradients, for second-order sensitivities, or in
        residual mode, the relative calculator computes the objective and
        its derivatives itself, with the inner parameters fixed at their
        optimal values, so the collector does not simulate at all. Returns
        ``None`` when it has to.

        Value-only calls with adjoint sensitivities are simulated like any
        other: the relative calculator alone would evaluate only the
        observables of its inner problem, leaving out the quantitative ones.
        """
        if not (
            (
                1 in sensi_orders
                and amici_solver.get_sensitivity_method()
                == asd.SensitivityMethod.adjoint
            )
            or 2 in sensi_orders
            or mode == MODE_RES
        ):
            return None
        return self.inner_calculators[0](
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )

    def _simulate(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
    ) -> tuple[list[asd.ReturnDataView], dict | None]:
        """Simulate with the inner parameters at their dummy values.

        Returns the simulation results, together with a result dict to
        return unchanged when the simulations failed.
        """
        from amici.sim.sundials.petab.v1 import fill_in_parameters

        # get dimension of outer problem
        dim = len(x_ids)

        # initialize return values
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        amici_solver.set_sensitivity_order(sensi_order)

        # fill in parameters, we expect here a RunTimeWarning to occur
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The following problem parameters were not used:.*",
                category=RuntimeWarning,
            )
            fill_in_parameters(
                edatas=edatas,
                problem_parameters=x_dct,
                scaled_parameters=True,
                parameter_mapping=parameter_mapping,
                amici_model=amici_model,
            )

        # run amici simulation
        rdatas = asd.run_simulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != asd.AMICI_SUCCESS for rdata in rdatas):
            ret = {
                FVAL: nllh,
                GRAD: snllh,
                HESS: s2nllh,
                RES: res,
                SRES: sres,
                RDATAS: rdatas,
                SPLINE_KNOTS: None,
                INNER_PARAMETERS: None,
            }
            ret[FVAL] = np.inf
            # if the gradient was requested,
            # we need to provide some value for it
            if 1 in sensi_orders:
                ret[GRAD] = np.full(shape=len(x_ids), fill_value=np.nan)
            return rdatas, ret

        return rdatas, None

    def __call__(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
    ):
        """Perform the actual AMICI call.

        Called within the :func:`AmiciObjective.__call__` method.
        Calls all the inner calculators and combines the results.

        Parameters
        ----------
        x_dct:
            Parameters for which to compute function value and derivatives.
        sensi_orders:
            Tuple of requested sensitivity orders.
        mode:
            Call mode (function value or residual based).
        amici_model:
            The AMICI model.
        amici_solver:
            The AMICI solver.
        edatas:
            The experimental data.
        n_threads:
            Number of threads for AMICI call.
        x_ids:
            Ids of optimization parameters.
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        fim_for_hess:
            Whether to use the FIM (if available) instead of the Hessian (if
            requested).
        """

        if mode == MODE_RES and any(
            data_type in self.data_types
            for data_type in [ORDINAL, CENSORED, SEMIQUANTITATIVE]
        ):
            raise NotImplementedError(
                f"Mode {mode} is not implemented for ordinal, censored or semi-quantitative data. "
                "However, it can be used if the only non-quantitative data type is relative data."
            )

        if 2 in sensi_orders and any(
            data_type in self.data_types
            for data_type in [ORDINAL, CENSORED, SEMIQUANTITATIVE]
        ):
            raise ValueError(
                "Hessian and FIM are not implemented for ordinal, censored or semi-quantitative data. "
                "However, they can be used if the only non-quantitative data type is relative data."
            )

        if (
            amici_solver.get_sensitivity_method()
            == asd.SensitivityMethod.adjoint
            and any(
                data_type in self.data_types
                for data_type in [ORDINAL, CENSORED, SEMIQUANTITATIVE]
            )
        ):
            raise NotImplementedError(
                "Adjoint sensitivity analysis is not implemented for ordinal, censored or semi-quantitative data. "
                "However, it can be used if the only non-quantitative data type is relative data."
            )

        if (
            ret := self._direct_to_relative_calculator(
                x_dct=x_dct,
                sensi_orders=sensi_orders,
                mode=mode,
                amici_model=amici_model,
                amici_solver=amici_solver,
                edatas=edatas,
                n_threads=n_threads,
                x_ids=x_ids,
                parameter_mapping=parameter_mapping,
                fim_for_hess=fim_for_hess,
            )
        ) is not None:
            return filter_return_dict(ret)

        # the inner parameters are not known yet, so simulate with dummies
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(self.necessary_par_dummy_values)

        rdatas, failure = self._simulate(
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
        if failure is not None:
            return filter_return_dict(failure)

        return self._combine_inner_results(
            rdatas=rdatas,
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
            index_slices=self._index_slices,
        )


class InnerCalculatorCollectorPetabV2(InnerCalculatorCollector):
    """Class to collect inner calculators for PEtab v2 problems.

    PEtab v2 counterpart of :class:`InnerCalculatorCollector`, for use with
    :class:`pypesto.objective.amici.amici.AmiciPetabV2Objective`. Simulations
    are delegated to the :class:`amici.sim.sundials.petab.PetabSimulator`,
    which maps the PEtab problem parameters to the model parameters. Only
    relative (and quantitative) data are supported.

    Parameters
    ----------
    data_types:
        List of non-quantitative data types in the problem.
    petab_simulator:
        The PEtab simulator of the :class:`AmiciPetabV2Objective` this
        calculator belongs to.
    inner_options:
        Options for the inner problems and solvers.
    """

    def __init__(
        self,
        data_types: set[str],
        petab_simulator: amici.sim.sundials.petab.PetabSimulator,
        inner_options: dict,
    ):
        from ..objective.amici.amici_calculator import AmiciCalculatorPetabV2

        self.petab_simulator = petab_simulator
        #: plain (non-hierarchical) evaluation of the PEtab v2 problem
        self._evaluator = AmiciCalculatorPetabV2(petab_simulator)

        edatas = petab_simulator.exp_man.create_edatas()
        super().__init__(
            data_types=data_types,
            petab_problem=petab_simulator.exp_man.petab_problem,
            model=petab_simulator.model,
            edatas=edatas,
            inner_options=inner_options,
        )
        #: the ``ExpData`` objects the index slices are built against
        self._edatas = edatas

    @staticmethod
    def _get_noise_models(
        petab_problem: v2.Problem,
        model: AmiciModel,
    ) -> list[tuple[str, str]]:
        """See :meth:`InnerCalculatorCollector._get_noise_models`."""
        from petab.v2 import C as V2C

        noise_models = {
            V2C.NORMAL: (LIN, NORMAL),
            V2C.LAPLACE: (LIN, LAPLACE),
            V2C.LOG_NORMAL: (LOG, NORMAL),
            V2C.LOG_LAPLACE: (LOG, LAPLACE),
        }
        return [
            noise_models[petab_problem[observable_id].noise_distribution]
            for observable_id in model.get_observable_ids()
        ]

    @property
    def free_parameter_ids(self) -> set[str] | None:
        """IDs of the parameters that are free in the pyPESTO problem.

        See :class:`AmiciCalculatorPetabV2`.
        """
        return self._evaluator.free_parameter_ids

    @free_parameter_ids.setter
    def free_parameter_ids(self, value: set[str] | None) -> None:
        self._evaluator.free_parameter_ids = value

    def construct_inner_calculators(
        self,
        petab_problem: v2.Problem,
        model: AmiciModel,
        edatas: list[asd.ExpData],
        inner_options: dict,
    ):
        """Construct inner calculators for each data type."""
        self.necessary_par_dummy_values = {}

        if unsupported_data_types := self.data_types - {RELATIVE}:
            raise NotImplementedError(
                f"Data types {unsupported_data_types} are not yet supported "
                "for PEtab v2 problems."
            )

        if RELATIVE in self.data_types:
            relative_inner_problem = RelativeInnerProblem.from_petab_v2_amici(
                petab_problem, model, edatas
            )
            self.necessary_par_dummy_values.update(
                relative_inner_problem.get_dummy_values(scaled=True)
            )
            self.inner_calculators.append(
                RelativeAmiciCalculator(
                    inner_problem=relative_inner_problem,
                    # PEtab v2 is simulated through the PEtab simulator, not
                    #  the v1 parameter-mapping machinery of the base class
                    evaluator=self._evaluator,
                )
            )
            self.relative_observable_ids = (
                relative_inner_problem.get_relative_observable_ids()
            )

    def _simulate(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
    ) -> tuple[list[asd.ReturnDataView], dict | None]:
        """See :meth:`InnerCalculatorCollector._simulate`.

        The PEtab simulator handles the parameter mapping, so
        ``parameter_mapping`` is unused.
        """
        if self._index_slices is None:
            self._index_slices = petab_v2_index_slices(
                petab_problem=self.petab_simulator.exp_man.petab_problem,
                par_sim_ids=self.petab_simulator.model.get_free_parameter_ids(),
                edatas=self._edatas,
                par_opt_ids=x_ids,
            )
            for calculator in self.inner_calculators:
                calculator.index_slices = self._index_slices

        ret = self._evaluator(
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
        rdatas = ret[RDATAS]
        # if any simulation failed, meaningful inner parameters are unlikely
        if any(rdata.status != asd.AMICI_SUCCESS for rdata in rdatas):
            return rdatas, ret
        return rdatas, None


def calculate_quantitative_result(
    rdatas: list[asd.ReturnDataView],
    edatas: list[asd.ExpData],
    sensi_orders: tuple[int],
    mode: ModeType,
    quantitative_data_mask: list[np.ndarray],
    noise_models: Sequence[tuple[str, str]],
    dim: int,
    parameter_mapping: ParameterMapping,
    par_opt_ids: list[str],
    par_sim_ids: list[str],
    index_slices: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Calculate the function values from rdatas and return as dict.

    ``noise_models`` is the ``(transformation, distribution)`` of each model
    observable. ``index_slices`` maps the simulation sensitivities onto the
    optimization parameters per condition; derived from ``parameter_mapping``
    if not given.
    """
    nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
        sensi_orders, mode, dim
    )

    # transform experimental data
    edatas = [asd.ExpDataView(edata)["measurements"] for edata in edatas]

    if 1 in sensi_orders and index_slices is None:
        index_slices = [
            par_index_slices(par_opt_ids, par_sim_ids, m.map_sim_var)
            for m in parameter_mapping
        ]

    # the model observables of each noise model
    observable_masks = {
        noise_model: np.array([nm == noise_model for nm in noise_models])
        for noise_model in set(noise_models)
    }

    # iterate over simulation conditions
    for condition_ix, (rdata, edata, mask) in enumerate(
        zip(rdatas, edatas, quantitative_data_mask, strict=True)
    ):
        for noise_model, observable_mask in observable_masks.items():
            # `observable_mask` broadcasts over the time points
            data_mask = mask & observable_mask
            nllh_i, dsim_i, dsigma_i = _noise_model_terms(
                data=edata[data_mask],
                sim=rdata[AMICI_Y][data_mask],
                sigma=rdata[AMICI_SIGMAY][data_mask],
                noise_model=noise_model,
            )
            nllh += np.sum(nllh_i)

            # calculate the gradient if requested
            if 1 in sensi_orders:
                # sensitivities of observables and sigmas,
                #  shape (n_simulation_parameters, n_data)
                sy_i = np.moveaxis(rdata[AMICI_SY], 1, 0)[:, data_mask]
                ssigma_i = np.moveaxis(rdata[AMICI_SSIGMAY], 1, 0)[
                    :, data_mask
                ]
                gradient_i = np.sum(
                    sy_i * dsim_i + ssigma_i * dsigma_i, axis=1
                )
                par_sim_slice, par_opt_slice = index_slices[condition_ix]
                np.add.at(snllh, par_opt_slice, gradient_i[par_sim_slice])

    ret = {
        FVAL: nllh,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas,
    }
    return filter_return_dict(ret)


def _noise_model_terms(
    data: np.ndarray,
    sim: np.ndarray,
    sigma: np.ndarray,
    noise_model: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the negative log-likelihood of each data point.

    The same as AMICI's cost function for the noise distribution
    ``{transformation}-{distribution}`` of
    ``noise_model = (transformation, distribution)``, see
    :func:`amici.importers.utils.noise_distribution_to_cost_function`.

    Returns
    -------
    The negative log-likelihood of each data point, and its derivatives
    w.r.t. ``sim`` and ``sigma``.
    """
    transformation, distribution = noise_model
    # residual on the transformed scale, its derivative w.r.t. `sim`, and the
    #  Jacobian of the transformation of the data
    if transformation == LIN:
        residual = sim - data
        dresidual_dsim = 1.0
        jacobian_term = 0.0
    elif transformation == LOG:
        residual = np.log(sim) - np.log(data)
        dresidual_dsim = 1 / sim
        jacobian_term = np.log(data)
    elif transformation == LOG10:
        residual = np.log10(sim) - np.log10(data)
        dresidual_dsim = 1 / (sim * np.log(10))
        jacobian_term = np.log(data * np.log(10))
    else:
        raise NotImplementedError(
            f"Observable transformation `{transformation}` is not supported."
        )

    if distribution == NORMAL:
        nllh = (
            0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * (residual / sigma) ** 2
        )
        dnllh_dresidual = residual / sigma**2
        dnllh_dsigma = (1 - (residual / sigma) ** 2) / sigma
    elif distribution == LAPLACE:
        nllh = np.log(2 * sigma) + np.abs(residual) / sigma
        dnllh_dresidual = np.sign(residual) / sigma
        dnllh_dsigma = (1 - np.abs(residual) / sigma) / sigma
    else:
        raise NotImplementedError(
            f"Noise distribution `{distribution}` is not supported."
        )

    return (
        nllh + jacobian_term,
        dnllh_dresidual * dresidual_dsim,
        dnllh_dsigma,
    )

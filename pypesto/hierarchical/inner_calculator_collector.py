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
    AMICI_SSIGMAZ,
    AMICI_SY,
    AMICI_Y,
    CENSORED,
    FVAL,
    GRAD,
    HESS,
    INNER_PARAMETERS,
    METHOD,
    MODE_RES,
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
    add_sim_grad_to_opt_grad,
    filter_return_dict,
    init_return_values,
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

        self._known_least_squares_safe = False

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
    ) -> list[np.ndarray]:
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

        # If there is no quantitative data, return None
        if not all(mask.any() for mask in quantitative_data_mask):
            return None

        return quantitative_data_mask

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
        from amici.sim.sundials.petab.v1 import fill_in_parameters

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

        # if we're using adjoint sensitivity analysis or need second order
        # sensitivities or are in residual mode, we can do so if the only
        # non-quantitative data type is relative data. In this case, we
        # use the relative calculator directly.
        if (
            amici_solver.get_sensitivity_method()
            == asd.SensitivityMethod.adjoint
            or 2 in sensi_orders
            or mode == MODE_RES
        ):
            relative_calculator = self.inner_calculators[0]
            ret = relative_calculator(
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
            return filter_return_dict(ret)

        # get dimension of outer problem
        dim = len(x_ids)

        # initialize return values
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        spline_knots = None
        interpretable_inner_pars = []

        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        amici_solver.set_sensitivity_order(sensi_order)

        x_dct = copy.deepcopy(x_dct)
        x_dct.update(self.necessary_par_dummy_values)
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
            return filter_return_dict(ret)

        if (
            not self._known_least_squares_safe
            and mode == MODE_RES
            and 1 in sensi_orders
        ):
            if not amici_model.get_add_sigma_residuals() and any(
                (
                    (r[AMICI_SSIGMAY] is not None and np.any(r[AMICI_SSIGMAY]))
                    or (
                        r[AMICI_SSIGMAZ] is not None
                        and np.any(r[AMICI_SSIGMAZ])
                    )
                )
                for r in rdatas
            ):
                raise RuntimeError(
                    "Cannot use least squares solver with"
                    "parameter dependent sigma! Support can be "
                    "enabled via "
                    "amici_model.set_add_sigma_residuals()."
                )
            self._known_least_squares_safe = True  # don't check this again

        # call inner calculators and collect results
        for calculator in self.inner_calculators:
            inner_result = calculator(
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
                rdatas=rdatas,
            )
            nllh += inner_result[FVAL]
            if 1 in sensi_orders:
                snllh += inner_result[GRAD]

            inner_pars = inner_result.get(INNER_PARAMETERS)
            if inner_pars is not None:
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
                dim=dim,
                parameter_mapping=parameter_mapping,
                par_opt_ids=x_ids,
                par_sim_ids=amici_model.get_free_parameter_ids(),
            )
            nllh += quantitative_result[FVAL]
            if 1 in sensi_orders:
                snllh += quantitative_result[GRAD]

        ret = {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas,
        }

        ret[INNER_PARAMETERS] = (
            interpretable_inner_pars
            if len(interpretable_inner_pars) > 0
            else None
        )
        ret[SPLINE_KNOTS] = spline_knots

        return filter_return_dict(ret)


class InnerCalculatorCollectorPetabV2(InnerCalculatorCollector):
    """Class to collect inner calculators for PEtab v2 problems.

    PEtab v2 counterpart of :class:`InnerCalculatorCollector`, for use with
    :class:`pypesto.objective.amici.amici.AmiciPetabV2Objective`. Instead of
    filling PEtab v1 style parameter mappings into the ``ExpData`` objects,
    simulations are delegated to the
    :class:`amici.sim.sundials.petab.PetabSimulator`, which handles the
    mapping of PEtab problem parameters to model parameters.

    Currently, only relative (and quantitative) data are supported. Ordinal,
    censored and semiquantitative data are not yet supported for PEtab v2.

    Parameters
    ----------
    data_types:
        List of non-quantitative data types in the problem.
    petab_simulator:
        The PEtab simulator that is used to simulate the (preprocessed) PEtab
        problem. Shared with the :class:`AmiciPetabV2Objective` that this
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
        # performs plain (non-hierarchical) evaluations of the PEtab v2
        #  problem; used whenever AMICI itself computes the objective value
        #  and derivatives, with the inner parameters fixed at their optima
        self._petab_v2_calculator = AmiciCalculatorPetabV2(petab_simulator)

        edatas = petab_simulator.exp_man.create_edatas()
        super().__init__(
            data_types=data_types,
            petab_problem=petab_simulator.exp_man.petab_problem,
            model=petab_simulator.model,
            edatas=edatas,
            inner_options=inner_options,
        )
        self._parameter_mapping = self._create_parameter_mapping(edatas)

    @property
    def free_parameter_ids(self) -> set[str] | None:
        """IDs of the parameters that are free in the pyPESTO problem.

        See :class:`AmiciCalculatorPetabV2`.
        """
        return self._petab_v2_calculator.free_parameter_ids

    @free_parameter_ids.setter
    def free_parameter_ids(self, value: set[str] | None) -> None:
        self._petab_v2_calculator.free_parameter_ids = value

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
            relative_inner_solver = RelativeAmiciCalculator(
                inner_problem=relative_inner_problem
            )
            self.inner_calculators.append(relative_inner_solver)
            self.relative_observable_ids = (
                relative_inner_problem.get_relative_observable_ids()
            )

    def _create_parameter_mapping(
        self, edatas: list[asd.ExpData]
    ) -> ParameterMapping:
        """Create a PEtab v1 style parameter mapping for gradient computation.

        The inner calculators map simulation gradients -- one entry per
        parameter for which sensitivities were computed (``ExpData.plist``) --
        to the optimization parameters via
        :func:`pypesto.objective.amici.amici_util.add_sim_grad_to_opt_grad`,
        which requires a PEtab v1 style per-condition parameter mapping. For
        PEtab v2, model parameters carry the IDs of the respective PEtab
        problem parameters, except for observable/noise placeholders, which
        are mapped per experiment.

        The mapping constructed here assigns to each model parameter the ID of
        the PEtab problem parameter it takes its value from, if sensitivities
        are computed for it (i.e., it is in ``ExpData.plist``), and a dummy
        numeric value otherwise (numeric entries indicate to
        ``add_sim_grad_to_opt_grad`` that there are no sensitivities).
        """
        from amici.sim._parameter_mapping import (
            ParameterMapping,
            ParameterMappingForCondition,
        )

        petab_problem = self.petab_simulator.exp_man.petab_problem
        model_par_ids = self.petab_simulator.model.get_free_parameter_ids()

        parameter_mapping = ParameterMapping()
        for edata, experiment in zip(
            edatas, petab_problem.experiments, strict=True
        ):
            if edata.id != experiment.id:
                raise AssertionError(
                    "The order of the ExpData objects does not match the "
                    "order of the PEtab experiments."
                )
            placeholder_mapping = _get_experiment_placeholder_mapping(
                petab_problem, experiment
            )
            plist = set(edata.plist)
            map_sim_var = {
                model_par_id: (
                    placeholder_mapping.get(model_par_id, model_par_id)
                    if model_par_ix in plist
                    else 0.0
                )
                for model_par_ix, model_par_id in enumerate(model_par_ids)
            }
            parameter_mapping.append(
                ParameterMappingForCondition(map_sim_var=map_sim_var)
            )
        return parameter_mapping

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
        """Perform the actual AMICI call, with hierarchical optimization.

        Called within the :func:`AmiciObjective.__call__` method. Calls all
        the inner calculators and combines the results.

        See :meth:`InnerCalculatorCollector.__call__` for the parameters.
        ``edatas`` and ``parameter_mapping`` are ignored: the PEtab simulator
        creates its own ``ExpData`` objects, and the parameter mapping was
        constructed upon initialization.
        """
        # get dimension of outer problem
        dim = len(x_ids)

        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        # a parameter that is free in the pyPESTO problem, but not estimated
        #  in the PEtab problem, has no sensitivities
        #  (see :class:`AmiciCalculatorPetabV2`)
        if self.free_parameter_ids is not None and sensi_order > 0:
            if missing := self.free_parameter_ids - set(
                self.petab_simulator.exp_man.petab_problem.x_free_ids
            ):
                raise ValueError(
                    f"Cannot compute gradient, missing entry for {missing}. "
                    "Those parameters are not estimated in the PEtab problem "
                    "-- fix them in the pyPESTO problem, or request "
                    "`sensi_orders=(0,)` only."
                )

        x_dct = copy.deepcopy(x_dct)
        x_dct.update(self.necessary_par_dummy_values)

        # If we are using adjoint sensitivity analysis, need second-order
        #  sensitivities, or are in residual mode, the objective function and
        #  its derivatives are computed by AMICI directly, with the inner
        #  parameters fixed at their optimal values.
        if (
            amici_solver.get_sensitivity_method()
            == asd.SensitivityMethod.adjoint
            or 2 in sensi_orders
            or mode == MODE_RES
        ):
            return self._call_amici_twice(
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

        # initialize return values
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        interpretable_inner_pars = []

        amici_solver.set_sensitivity_order(sensi_order)

        # run amici simulation; the mapping of the (dummy-augmented) PEtab
        #  problem parameters to model parameters is handled by the PEtab
        #  simulator
        result = self.petab_simulator.simulate(x_dct)
        rdatas = result.rdatas
        # the simulator creates its own ExpData objects
        edatas = result.edatas

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != asd.AMICI_SUCCESS for rdata in rdatas):
            ret = {
                FVAL: np.inf,
                GRAD: snllh,
                HESS: s2nllh,
                RES: res,
                SRES: sres,
                RDATAS: rdatas,
                SPLINE_KNOTS: None,
                INNER_PARAMETERS: None,
            }
            # if the gradient was requested,
            # we need to provide some value for it
            if 1 in sensi_orders:
                ret[GRAD] = np.full(shape=len(x_ids), fill_value=np.nan)
            return filter_return_dict(ret)

        # call inner calculators and collect results
        for calculator in self.inner_calculators:
            inner_result = calculator(
                x_dct=x_dct,
                sensi_orders=sensi_orders,
                mode=mode,
                amici_model=amici_model,
                amici_solver=amici_solver,
                edatas=edatas,
                n_threads=n_threads,
                x_ids=x_ids,
                parameter_mapping=self._parameter_mapping,
                fim_for_hess=fim_for_hess,
                rdatas=rdatas,
            )
            nllh += inner_result[FVAL]
            if 1 in sensi_orders:
                snllh += inner_result[GRAD]

            inner_pars = inner_result.get(INNER_PARAMETERS)
            if inner_pars is not None:
                interpretable_inner_pars.extend(inner_pars)

        # add the quantitative data contribution
        if self.quantitative_data_mask is not None:
            quantitative_result = calculate_quantitative_result(
                rdatas=rdatas,
                sensi_orders=sensi_orders,
                edatas=edatas,
                mode=mode,
                quantitative_data_mask=self.quantitative_data_mask,
                dim=dim,
                parameter_mapping=self._parameter_mapping,
                par_opt_ids=x_ids,
                par_sim_ids=amici_model.get_free_parameter_ids(),
            )
            nllh += quantitative_result[FVAL]
            if 1 in sensi_orders:
                snllh += quantitative_result[GRAD]

        ret = {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas,
        }

        ret[INNER_PARAMETERS] = (
            interpretable_inner_pars
            if len(interpretable_inner_pars) > 0
            else None
        )
        ret[SPLINE_KNOTS] = None

        return filter_return_dict(ret)

    def _call_amici_twice(
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
        """Calculate by calling AMICI twice.

        This is necessary if the adjoint method is used, or if the Hessian or
        residuals are requested. In these cases, AMICI is called first to
        obtain simulations for the calculation of the optimal inner
        parameters, and then again to obtain the requested quantities, with
        the inner parameters fixed at their optimal values.

        ``x_dct`` is expected to already contain dummy values for the inner
        parameters.
        """
        dim = len(x_ids)
        relative_calculator = self.inner_calculators[0]
        inner_problem = relative_calculator.inner_problem

        # simulate without sensitivities to compute the optimal inner
        #  parameters
        inner_result = self._petab_v2_calculator(
            x_dct=x_dct,
            sensi_orders=(0,),
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
        rdatas = inner_result[RDATAS]

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != asd.AMICI_SUCCESS for rdata in rdatas):
            # if the gradient was requested, we need to provide some value
            # for it
            if 1 in sensi_orders:
                inner_result[GRAD] = np.full(shape=dim, fill_value=np.nan)
            if 2 in sensi_orders:
                inner_result[HESS] = np.full(
                    shape=(dim, dim), fill_value=np.nan
                )
            return filter_return_dict(inner_result)

        inner_parameters = relative_calculator.inner_solver.solve(
            problem=inner_problem,
            sim=[rdata[AMICI_Y] for rdata in rdatas],
            sigma=[rdata[AMICI_SIGMAY] for rdata in rdatas],
            scaled=True,
        )

        # compute the requested quantities with the inner parameters fixed
        #  at their optimal values
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(inner_parameters)

        ret = self._petab_v2_calculator(
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
        ret[INNER_PARAMETERS] = np.array(
            [inner_parameters[x_id] for x_id in inner_problem.get_x_ids()]
        )
        return filter_return_dict(ret)


def _get_experiment_placeholder_mapping(
    petab_problem: v2.Problem,
    experiment: v2.Experiment,
) -> dict[str, str]:
    """Get the placeholder mapping for the given experiment.

    Maps model parameter IDs (= PEtab observable/noise placeholder IDs) to
    the IDs of the PEtab problem parameters that override them in the given
    experiment. Numeric overrides are omitted.

    Because AMICI does not support timepoint-specific overrides, this mapping
    is unique (see
    :meth:`amici.sim.sundials.petab.ExperimentManager.apply_parameters`).
    """
    observables = {
        observable.id: observable for observable in petab_problem.observables
    }
    mapping = {}
    for measurement in petab_problem.get_measurements_for_experiment(
        experiment
    ):
        observable = observables[measurement.observable_id]
        for placeholder, override in zip(
            observable.observable_placeholders,
            measurement.observable_parameters,
            strict=True,
        ):
            if override.is_Symbol:
                mapping[str(placeholder)] = str(override)
        for placeholder, override in zip(
            observable.noise_placeholders,
            measurement.noise_parameters,
            strict=True,
        ):
            if override.is_Symbol:
                mapping[str(placeholder)] = str(override)
    return mapping


def calculate_quantitative_result(
    rdatas: list[asd.ReturnDataView],
    edatas: list[asd.ExpData],
    sensi_orders: tuple[int],
    mode: ModeType,
    quantitative_data_mask: list[np.ndarray],
    dim: int,
    parameter_mapping: ParameterMapping,
    par_opt_ids: list[str],
    par_sim_ids: list[str],
):
    """Calculate the function values from rdatas and return as dict."""
    nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
        sensi_orders, mode, dim
    )

    # transform experimental data
    edatas = [asd.ExpDataView(edata)["measurements"] for edata in edatas]

    # calculate the function value
    for rdata, edata, mask in zip(
        rdatas, edatas, quantitative_data_mask, strict=True
    ):
        data_i = edata[mask]
        sim_i = rdata[AMICI_Y][mask]
        sigma_i = rdata[AMICI_SIGMAY][mask]

        nllh += 0.5 * np.nansum(
            np.log(2 * np.pi * sigma_i**2) + (data_i - sim_i) ** 2 / sigma_i**2
        )

    # calculate the gradient if requested
    if 1 in sensi_orders:
        parameter_map_sim_var = [
            cond_par_map.map_sim_var for cond_par_map in parameter_mapping
        ]
        # iterate over simulation conditions
        for (
            rdata,
            edata,
            mask,
            condition_map_sim_var,
        ) in zip(
            rdatas,
            edatas,
            quantitative_data_mask,
            parameter_map_sim_var,
            strict=True,
        ):
            data_i = edata[mask]
            sim_i = rdata[AMICI_Y][mask]
            sigma_i = rdata[AMICI_SIGMAY][mask]

            n_parameters = rdata[AMICI_SY].shape[1]

            # Get sensitivities of observables and sigmas
            sensitivities_i = np.asarray(
                [
                    rdata[AMICI_SY][:, parameter_index, :][mask]
                    for parameter_index in range(n_parameters)
                ]
            )
            ssigma_i = np.asarray(
                [
                    rdata[AMICI_SSIGMAY][:, parameter_index, :][mask]
                    for parameter_index in range(n_parameters)
                ]
            )
            # calculate the gradient for the condition
            gradient_for_condition = np.nansum(
                np.multiply(
                    ssigma_i,
                    (
                        (
                            np.full(len(data_i), 1)
                            - (data_i - sim_i) ** 2 / sigma_i**2
                        )
                        / sigma_i
                    ),
                ),
                axis=1,
            ) + np.nansum(
                np.multiply(sensitivities_i, ((sim_i - data_i) / sigma_i**2)),
                axis=1,
            )
            add_sim_grad_to_opt_grad(
                par_opt_ids=par_opt_ids,
                par_sim_ids=par_sim_ids,
                condition_map_sim_var=condition_map_sim_var,
                sim_grad=gradient_for_condition,
                opt_grad=snllh,
            )

    ret = {
        FVAL: nllh,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas,
    }
    return filter_return_dict(ret)

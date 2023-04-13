"""Module for the InnerCalculatorCollector class.

In case of non-quantitative measurements, this class is used to collect
hierarchical inner calculators for each data type and merge their results.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

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
    MEASUREMENT_TYPE,
    METHOD,
    MODE_RES,
    NONLINEAR_MONOTONE,
    OPTIMAL_SCALING_OPTIONS,
    ORDINAL,
    RDATAS,
    RES,
    SPLINE_APPROXIMATION_OPTIONS,
    SPLINE_RATIO,
    SRES,
    X_INNER_OPT,
    ModeType,
)
from ..objective.amici.amici_calculator import AmiciCalculator
from ..objective.amici.amici_util import filter_return_dict, init_return_values

try:
    import amici
    import petab
    from amici.parameter_mapping import ParameterMapping
    from petab.C import OBSERVABLE_ID
except ImportError:
    petab = None
    ParameterMapping = None

from .optimal_scaling import (
    OptimalScalingAmiciCalculator,
    OptimalScalingInnerSolver,
    OptimalScalingProblem,
)
from .spline_approximation import (
    SplineAmiciCalculator,
    SplineInnerProblem,
    SplineInnerSolver,
)

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


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
        data_types: List[str],
        petab_problem: 'petab.Problem',
        model: AmiciModel,
        edatas: List['amici.ExpData'],
        inner_options: Dict,
    ):
        super().__init__()
        self.validate_options(inner_options)

        self.data_types = data_types
        self.inner_calculators = []
        self.construct_inner_calculators(
            petab_problem, model, edatas, inner_options
        )

        self.quantitative_data_mask = self._get_quantitative_data_mask(
            petab_problem, model.getObservableIds(), edatas
        )

        self._known_least_squares_safe = False

    def initialize(self):
        """Initialize."""
        for calculator in self.inner_calculators:
            calculator.initialize()

    def construct_inner_calculators(
        self,
        petab_problem: 'petab.Problem',
        model: AmiciModel,
        edatas: List['amici.ExpData'],
        inner_options: Dict,
    ):
        """Construct inner calculators for each data type."""
        if ORDINAL in self.data_types or CENSORED in self.data_types:
            optimal_scaling_inner_options = {
                key: value
                for key, value in inner_options.items()
                if key in OPTIMAL_SCALING_OPTIONS
            }
            inner_problem_method = optimal_scaling_inner_options.get(
                METHOD, None
            )
            os_inner_problem = OptimalScalingProblem.from_petab_amici(
                petab_problem, model, edatas, inner_problem_method
            )
            os_inner_solver = OptimalScalingInnerSolver(
                options=optimal_scaling_inner_options
            )
            os_calculator = OptimalScalingAmiciCalculator(
                os_inner_problem, os_inner_solver
            )
            self.inner_calculators.append(os_calculator)

        if NONLINEAR_MONOTONE in self.data_types:
            spline_inner_options = {
                key: value
                for key, value in inner_options.items()
                if key in SPLINE_APPROXIMATION_OPTIONS
            }
            spline_ratio = spline_inner_options.pop(SPLINE_RATIO, None)
            spline_inner_problem = SplineInnerProblem.from_petab_amici(
                petab_problem, model, edatas, spline_ratio
            )
            spline_inner_solver = SplineInnerSolver(
                options=spline_inner_options
            )
            spline_calculator = SplineAmiciCalculator(
                spline_inner_problem, spline_inner_solver
            )
            self.inner_calculators.append(spline_calculator)
        # TODO relative data

        if set(self.data_types) - {ORDINAL, CENSORED, NONLINEAR_MONOTONE}:
            unsupported_data_types = set(self.data_types) - {
                ORDINAL,
                CENSORED,
                NONLINEAR_MONOTONE,
            }
            raise NotImplementedError(
                f"Data types {unsupported_data_types} are not supported."
            )

    def validate_options(self, inner_options: Dict):
        """Validate the inner options.

        Parameters
        ----------
        inner_options:
            Options for the inner problems and solvers.
        """
        for key in inner_options:
            if (
                key not in OPTIMAL_SCALING_OPTIONS
                and key not in SPLINE_APPROXIMATION_OPTIONS
            ):
                raise ValueError(f"Unknown inner option {key}.")

    def _get_quantitative_data_mask(
        self,
        petab_problem: 'petab.Problem',
        observable_ids: List[str],
        edatas: List['amici.ExpData'],
    ) -> List[np.ndarray]:
        # transform experimental data
        edatas = [
            amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
        ]

        measurement_df = petab_problem.measurement_df

        quantitative_data_mask = [
            np.zeros_like(edata, dtype=bool) for edata in edatas
        ]

        for observable_idx, observable_id in enumerate(observable_ids):
            observable_df = measurement_df[
                measurement_df[OBSERVABLE_ID] == observable_id
            ]
            # If the MEASUREMENT_TYPE column is filled with nans,
            # then fill that axis of the mask with True
            if observable_df[MEASUREMENT_TYPE].isna().all():
                for condition_mask in quantitative_data_mask:
                    condition_mask[:, observable_idx] = True

        # If there is no quantitative data, return None
        if not all(mask.any() for mask in quantitative_data_mask):
            return None

        return quantitative_data_mask

    def __call__(
        self,
        x_dct: Dict,
        sensi_orders: Tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: List[amici.ExpData],
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
        import amici.parameter_mapping

        if mode == MODE_RES:
            raise NotImplementedError(
                f"Mode {mode} is not implemented for the :class:`pypesto.objective.amici.InnerCalculatorCollector`."
            )

        if 2 in sensi_orders:
            raise ValueError(
                "Hessian and FIM are not implemented for the :class:`pypesto.objective.amici.InnerCalculatorCollector`."
            )

        # get dimension of outer problem
        dim = len(x_ids)

        # initialize return values
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        inner_parameters_dictionary = {}

        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        amici_solver.setSensitivityOrder(sensi_order)

        x_dct = copy.deepcopy(x_dct)

        # fill in parameters
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model,
        )

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != amici.AMICI_SUCCESS for rdata in rdatas):
            ret = {
                FVAL: nllh,
                GRAD: snllh,
                HESS: s2nllh,
                RES: res,
                SRES: sres,
                RDATAS: rdatas,
                X_INNER_OPT: inner_parameters_dictionary,
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
            if not amici_model.getAddSigmaResiduals() and any(
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
                    'Cannot use least squares solver with'
                    'parameter dependent sigma! Support can be '
                    'enabled via '
                    'amici_model.setAddSigmaResiduals().'
                )
            self._known_least_squares_safe = True  # don't check this again

        # call inner calculators
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
            if sensi_order > 0:
                snllh += inner_result[GRAD]
            inner_parameters_dictionary.update(inner_result[X_INNER_OPT])

        # add result for quantitative data
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
                par_sim_ids=amici_model.getParameterIds(),
                par_edatas_indices=[edata.plist for edata in edatas],
            )
            nllh += quantitative_result[FVAL]
            if sensi_order > 0:
                snllh += quantitative_result[GRAD]

        ret = {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas,
            X_INNER_OPT: inner_parameters_dictionary,
        }

        return filter_return_dict(ret)


def calculate_quantitative_result(
    rdatas: List[amici.ReturnDataView],
    edatas: List[amici.ExpData],
    sensi_orders: Tuple[int],
    mode: ModeType,
    quantitative_data_mask: List[np.ndarray],
    dim: int,
    parameter_mapping: ParameterMapping,
    par_opt_ids: List[str],
    par_sim_ids: List[str],
    par_edatas_indices: List[List[int]],
):
    """Calculate the function values from rdatas and return as dict."""
    nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
        sensi_orders, mode, dim
    )

    # transform experimental data
    edatas = [
        amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
    ]

    # calculate the function value
    for rdata, edata, mask in zip(rdatas, edatas, quantitative_data_mask):
        data_i = edata[mask]
        sim_i = rdata[AMICI_Y][mask]
        sigma_i = rdata[AMICI_SIGMAY][mask]

        nllh += 0.5 * np.nansum(
            np.log(2 * np.pi * sigma_i**2)
            + (data_i - sim_i) ** 2 / sigma_i**2
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
            par_edata_indices,
        ) in zip(
            rdatas,
            edatas,
            quantitative_data_mask,
            parameter_map_sim_var,
            par_edatas_indices,
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
                    rdata[AMICI_SSIGMAY][:, parameter_index][mask]
                    for parameter_index in range(n_parameters)
                ]
            )
            # calculate the gradient for the condition
            gradient_for_condition = ssigma_i @ (
                (
                    np.full(len(data_i), 1)
                    - (data_i - sim_i) ** 2 / sigma_i**2
                )
                / sigma_i
            ) - sensitivities_i @ ((data_i - sim_i) / sigma_i**2)

            # add gradient to correct index of snllh
            for par_sim, par_opt in condition_map_sim_var.items():
                if not isinstance(par_opt, str):
                    continue
                par_opt_idx = par_opt_ids.index(par_opt)
                par_sim_idx = par_sim_ids.index(par_sim)
                par_edata_idx = (
                    par_edata_indices.index(par_sim_idx)
                    if par_sim_idx in par_edata_indices
                    else None
                )

                if par_edata_idx is not None:
                    snllh[par_opt_idx] += gradient_for_condition[par_edata_idx]

    ret = {
        FVAL: nllh,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas,
    }
    return filter_return_dict(ret)

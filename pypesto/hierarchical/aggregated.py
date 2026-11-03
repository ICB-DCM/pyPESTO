"""Aggregation of objectives with independent inner problems."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..C import INNER_PARAMETERS
from ..objective import AggregatedObjective
from ..objective.base import ResultDict
from .inner_calculator_collector import InnerCalculatorCollector


class AggregatedInnerCalculators:
    """The inner calculators of several independent collectors.

    Provides the inner parameter information of all collectors like a
    single :class:`InnerCalculatorCollector`, in the order of the collectors.

    Parameters
    ----------
    collectors:
        The collectors.
    """

    def __init__(self, collectors: Sequence[InnerCalculatorCollector]):
        self.inner_calculators = [
            calculator
            for collector in collectors
            for calculator in collector.inner_calculators
        ]
        self.semiquant_observable_ids = [
            observable_id
            for collector in collectors
            for observable_id in collector.semiquant_observable_ids or []
        ] or None
        self.relative_observable_ids = [
            observable_id
            for collector in collectors
            for observable_id in collector.relative_observable_ids or []
        ] or None

    # these only use `inner_calculators`
    get_inner_par_ids = InnerCalculatorCollector.get_inner_par_ids
    get_interpretable_inner_par_ids = (
        InnerCalculatorCollector.get_interpretable_inner_par_ids
    )
    get_interpretable_inner_par_bounds = (
        InnerCalculatorCollector.get_interpretable_inner_par_bounds
    )
    get_interpretable_inner_par_scales = (
        InnerCalculatorCollector.get_interpretable_inner_par_scales
    )


class AggregatedHierarchicalObjective(AggregatedObjective):
    """Sum of objectives whose inner problems are independent.

    Such as the per-model objectives of a multi-model PEtab v2 problem
    whose models do not share inner parameters. Objectives without inner
    problems are allowed.

    The inner parameters are those of the objectives, in the order of the
    objectives, both in :attr:`calculator` and in the returned
    ``INNER_PARAMETERS``.
    """

    @property
    def calculator(self) -> AggregatedInnerCalculators:
        """The inner calculators of all objectives."""
        return AggregatedInnerCalculators(
            [
                objective.calculator
                for objective in self._objectives
                if isinstance(
                    getattr(objective, "calculator", None),
                    InnerCalculatorCollector,
                )
            ]
        )

    def _aggregate_results(self, rvals: Sequence[ResultDict]) -> ResultDict:
        """Aggregate the results, and concatenate the inner parameters."""
        result = super()._aggregate_results(rvals)
        if all(rval.get(INNER_PARAMETERS) is None for rval in rvals):
            return result
        # an objective whose simulation failed returns no inner parameters;
        #  NaNs keep the others under their names
        result[INNER_PARAMETERS] = [
            x
            for objective, rval in zip(self._objectives, rvals, strict=True)
            if isinstance(
                calculator := getattr(objective, "calculator", None),
                InnerCalculatorCollector,
            )
            for x in (
                rval[INNER_PARAMETERS]
                if rval.get(INNER_PARAMETERS) is not None
                else [np.nan]
                * len(calculator.get_interpretable_inner_par_ids())
            )
        ]
        return result

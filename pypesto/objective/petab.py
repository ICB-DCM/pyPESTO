"""Objective function for PEtab models using the PEtab simulator."""

from __future__ import annotations

try:
    import petab.v1 as petab
    from petab.v1.simulate import Simulator as PetabSimulator
except ImportError:
    petab = None
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from ..C import FVAL, MODE_FUN, MODE_RES, RES, ModeType
from .base import ObjectiveBase, ResultDict


class PetabSimulatorObjective(ObjectiveBase):
    """Objective function for PEtab models using the PEtab simulator."""

    def __init__(
        self,
        simulator: PetabSimulator,
        x_names: Sequence[str] | None = None,
    ):
        """Initialize the PEtab simulator objective function.

        Parameters
        ----------
        petab_problem:
            The PEtab problem.
        simulator:
            The PEtab simulator.
        x_names:
            Names of optimization parameters.
        """
        if petab is None:
            raise ImportError(
                "The `petab` package is required for this objective function."
            )
        self.simulator = simulator
        self.petab_problem = self.simulator.petab_problem
        if x_names is None:
            x_names = list(self.petab_problem.get_x_ids())
        super().__init__(x_names=x_names)

    def replace_parameters(self, x: np.ndarray):
        """Replace the parameters in the PEtab problem with the given values.

        Parameters
        ----------
        x:
            Parameter vector for optimization.
        """
        x_dict = OrderedDict(zip(self._x_names, x))
        x_unscaled = self.petab_problem.unscale_parameters(x_dict)
        par_df = self.petab_problem.parameter_df
        par_df["nominalValue"] = par_df.index.map(x_unscaled)
        self.simulator.set_parameters(x_unscaled)

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        return_dict: bool,
        **kwargs,
    ) -> ResultDict:
        """See :meth:`ObjectiveBase.call_unprocessed`."""

        self.replace_parameters(x)
        sim_df = self.simulator.simulate(noise=False, as_measurement=False)
        result = {}
        result["simulations"] = sim_df
        if mode == MODE_FUN:
            result[FVAL] = -petab.calculate_llh(
                measurement_dfs=self.petab_problem.measurement_df,
                simulation_dfs=sim_df,
                observable_dfs=self.petab_problem.observable_df,
                parameter_dfs=self.petab_problem.parameter_df,
            )
        elif mode == MODE_RES:
            result[RES] = petab.calculate_residuals(
                measurement_dfs=self.petab_problem.measurement_df,
                simulation_dfs=sim_df,
                observable_dfs=self.petab_problem.observable_df,
                parameter_dfs=self.petab_problem.parameter_df,
            )
        return result

    def check_sensi_orders(
        self,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
    ) -> bool:
        """See :class:`ObjectiveBase` documentation."""
        if not sensi_orders:
            return True
        sensi_order = max(sensi_orders)
        max_sensi_order = 0

        return sensi_order <= max_sensi_order

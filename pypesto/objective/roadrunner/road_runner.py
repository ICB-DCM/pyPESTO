"""Objective function for RoadRunner models.

Currently does not support sensitivities.
"""
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import roadrunner
from petab import Problem as PetabProblem
from petab.parameter_mapping import ParMappingDictQuadruple

from ...C import MODE_FUN, MODE_RES, ROADRUNNER_INSTANCE, X_NAMES, ModeType
from ..base import ObjectiveBase
from .roadrunner_calculator import RoadRunnerCalculator
from .utils import ExpData, SolverOptions


class RoadRunnerObjective(ObjectiveBase):
    """Objective function for RoadRunner models.

    Currently does not support sensitivities.
    """

    def __init__(
        self,
        rr: roadrunner.RoadRunner,
        edatas: Union[Sequence[ExpData], ExpData],
        parameter_mapping: list[ParMappingDictQuadruple],
        petab_problem: PetabProblem,
        calculator: Optional[RoadRunnerCalculator] = None,
        x_names: Optional[Sequence[str]] = None,
        solver_options: Optional[SolverOptions] = None,
    ):
        """Initialize the RoadRunner objective function.

        Parameters
        ----------
        rr:
            RoadRunner instance for simulation.
        edatas:
            The experimental data. If a list is passed, its entries correspond
            to multiple experimental conditions.
        parameter_mapping:
            Mapping of optimization parameters to model parameters. Format as
            created by `petab.get_optimization_to_simulation_parameter_mapping`.
            The default is just to assume that optimization and simulation
            parameters coincide.
        petab_problem:
            The corresponding PEtab problem. Needed to calculate NLLH.
            Might be removed later.
        calculator:
            The calculator to use. If None, a new instance is created.
        x_names:
            Names of optimization parameters.
        """
        self.roadrunner_instance = rr
        # make sure edatas are a list
        if isinstance(edatas, ExpData):
            edatas = [edatas]
        self.edatas = edatas
        self.parameter_mapping = parameter_mapping
        self.petab_problem = petab_problem
        if calculator is None:
            calculator = RoadRunnerCalculator()
        self.calculator = calculator
        if solver_options is None:
            solver_options = SolverOptions()
        self.solver_options = solver_options
        super().__init__(x_names=x_names)

    def get_config(self) -> dict:
        """Return basic information of the objective configuration."""
        info = super().get_config()
        info["solver_options"] = repr(self.solver_options)
        info[X_NAMES] = self.x_names
        info[ROADRUNNER_INSTANCE] = self.roadrunner_instance.getInfo()
        return info

    # TODO: Check whether we need some sort of pickling

    def __call__(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...] = (0,),
        mode: ModeType = MODE_FUN,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray, dict]:
        """See :class:`ObjectiveBase` documentation."""
        return super().__call__(x, sensi_orders, mode, return_dict, **kwargs)

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        edatas: Optional[Sequence[ExpData]] = None,
        parameter_mapping: Optional[list[ParMappingDictQuadruple]] = None,
    ) -> dict:
        """
        Call objective function without pre- or post-processing and formatting.

        Returns
        -------
        result:
            A dict containing the results.
        """
        # fill in values if not passed
        if edatas is None:
            edatas = self.edatas
        if parameter_mapping is None:
            parameter_mapping = self.parameter_mapping
        # convert x to dictionary
        x = OrderedDict(zip(self.x_names, x))
        ret = self.calculator(
            x_dct=x,
            mode=mode,
            roadrunner_instance=self.roadrunner_instance,
            edatas=edatas,
            x_ids=self.x_names,
            parameter_mapping=parameter_mapping,
            petab_problem=self.petab_problem,
            solver_options=self.solver_options,
        )
        return ret

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

    def check_mode(self, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        return mode in [MODE_FUN, MODE_RES]

import numpy as np
from copy import deepcopy
from typing import List
from .objective import Objective

from .constants import RDATAS


class AggregatedObjective(Objective):
    """
    This class aggregates multiple objectives into one objective.
    """

    def __init__(
            self,
            objectives: List[Objective],
            x_names: List[str] = None):
        """
        Constructor.

        Parameters
        ----------
        objectives: list
            List of pypesto.objetive instances
        """
        # input typechecks
        if not isinstance(objectives, list):
            raise TypeError(f'Objectives must be a list, '
                            f'was {type(objectives)}.')

        if not all(
                isinstance(objective, Objective)
                for objective in objectives
        ):
            raise TypeError('Objectives must only contain elements of type'
                            'pypesto.Objective')

        if not len(objectives):
            raise ValueError('Length of objectives must be at least one')

        self.objectives = objectives

        # assemble a dict that we can pass as kwargs to the
        # pypesto.Objective constructor
        init_kwargs = {
            'x_names': x_names
        }

        # check if all objectives consistently accept sensi orders in fun/res
        # and adopt the same behaviour in aggregate
        for attr in ['fun_accept_sensi_orders', 'res_accept_sensi_orders']:
            _check_boolean_value_consistent(objectives, attr)
            init_kwargs[attr] = getattr(objectives[0], attr)

        # check if all objectives consistently accept sensi orders in fun/res
        # and adopt the same behaviour in aggregate
        for attr in [
            'fun', 'grad', 'hess', 'hessp', 'res', 'sres'
        ]:
            if any(
                getattr(objective, attr) is None
                for objective in objectives
            ):
                _check_none_value_consistent(objectives, attr)
                init_kwargs[attr] = None
            elif all(
                isinstance(getattr(objective, attr), bool)
                for objective in objectives
            ):
                _check_boolean_value_consistent(objectives, attr)
                init_kwargs[attr] = getattr(objectives[0], attr)
            elif all(
                callable(getattr(objective, attr))
                for objective in objectives
            ):
                aggregate_fun = f'aggregate_{attr}'
                if (
                        attr == 'fun'
                        and init_kwargs['fun_accept_sensi_orders']
                ) or (
                        attr == 'res'
                        and init_kwargs['res_accept_sensi_orders']
                ):
                    aggregate_fun += '_sensi_orders'

                init_kwargs[attr] = getattr(self, aggregate_fun)
            else:
                raise ValueError(f'{attr} has incompatible types across '
                                 f'instances!')

        super().__init__(**init_kwargs)

    def __deepcopy__(self, memodict=None):
        other = AggregatedObjective(
            objectives=[deepcopy(objective) for objective in self.objectives],
            x_names=deepcopy(self.x_names),
        )
        return other

    def aggregate_fun_sensi_orders(self, x, sensi_orders):
        rvals = [
            objective.fun(x, sensi_orders)
            for objective in self.objectives
        ]

        # sum over fval/grad/hess
        result = {
            key: sum(rval[key] for rval in rvals)
            for key in ['fval', 'grad', 'hess']
            if key in rvals[0]
        }

        # extract rdatas and flatten
        result[RDATAS] = []
        for rval in rvals:
            if RDATAS in rval:
                result[RDATAS].extend(rval[RDATAS])

        return result

    def aggregate_res_sensi_orders(self, x, sensi_orders):
        result = dict()

        # initialize res and sres
        rval0 = self.objectives[0].res(x, sensi_orders)
        if 'res' in rval0:
            res = np.asarray(rval0['res'])
        else:
            res = None

        if 'sres' in rval0:
            sres = np.asarray(rval0['sres'])
        else:
            sres = None

        if RDATAS in rval0:
            result[RDATAS] = rval0[RDATAS]
        else:
            result[RDATAS] = []

        # skip iobj=0 after initialization, stack matrices
        for iobj in range(1, len(self.objectives)):
            rval = self.objectives[iobj].res(x, sensi_orders)
            if res is not None:
                res = np.hstack([res, np.asarray(rval['res'])])
            if sres is not None:
                sres = np.vstack([sres, np.asarray(rval['sres'])])
            if RDATAS in rval:
                result[RDATAS].extend(rval[RDATAS])

        # transform results to dict

        if res is not None:
            result['res'] = res
        if sres is not None:
            result['sres'] = sres

        return result

    def aggregate_res(self, x):
        if self.sres is True:  # integrated mode
            res = self.objectives[0].res(x)[0]
        else:
            res = self.objectives[0].res(x)
        for iobj in range(1, len(self.objectives)):
            if self.sres is True:  # integrated mode
                res_stack = np.asarray(self.objectives[iobj].res(x))[0]
            else:
                res_stack = np.asarray(self.objectives[iobj].res(x))
            res = np.hstack([res, res_stack])

        return res

    def aggregate_sres(self, x):
        sres = self.objectives[0].sres(x)
        for iobj in range(1, len(self.objectives)):
            sres = np.vstack([sres, np.asarray(self.objectives[iobj].sres(x))])

        return sres

    def aggregate_fun(self, x):
        if self.grad is True:  # integrated mode
            return tuple(
                sum(objective.fun(x)[idx] for objective in self.objectives)
                for idx in range(2+self.hess)
            )
        else:
            return sum(objective.fun(x) for objective in self.objectives)

    def aggregate_grad(self, x):
        return sum(objective.grad(x) for objective in self.objectives)

    def aggregate_hess(self, x):
        return sum(objective.hess(x) for objective in self.objectives)

    def aggregate_hessp(self, x):
        return sum(objective.hessp(x) for objective in self.objectives)

    def reset_steadystate_guesses(self):
        """
        Propagates reset_steadystate_guesses() to child objectives if available
        (currently only applies for amici_objective)
        """
        for objective in self.objectives:
            if hasattr(objective, 'reset_steadystate_guesses'):
                objective.reset_steadystate_guesses()


def _check_boolean_value_consistent(objectives, attr):
    values = set(
        getattr(objective, attr)
        for objective in objectives
    )
    if len(values) > 1:
        raise ValueError(f'{attr} of all objectives must have a consistent '
                         f'value!')


def _check_none_value_consistent(objectives, attr):
    is_none = (
        getattr(objective, attr) is None
        for objective in objectives
    )
    if not all(is_none):
        raise ValueError(f'{attr} of all objectives must have a consistent '
                         f'value!')

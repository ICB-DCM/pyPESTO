import numpy as np

from .objective import Objective


class AggregateObjective(Objective):
    """
    This class allows to create an aggregateObjective from a list of
    Objective instances.
    """

    def __init__(self, objectives, x_names=None, options=None):
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

        self.objectives = objectives

        # assemble a dict that we can pass as kwargs to the
        # pypesto.Objective constructor
        init_kwargs = {
            'x_names': x_names,
            'options': options
        }

        # check if all objectives consistently accept sensi orders in fun/res
        # and adopt the same behaviour in aggregate
        for attr in ['fun_accept_sensi_orders', 'res_accept_sensi_orders']:
            _check_boolean_value_consistent(objectives, attr)
            init_kwargs[attr] = getattr(objectives[0], attr)

        # check if all objectives consistently accept sensi orders in fun/res
        # and adopt the same behaviour in aggregate
        for attributes in [
            'fun', 'grad', 'hess', 'hessp', 'res', 'sres'
        ]:
            if any(
                getattr(objective, attr) is None
                for objective in objectives
            ):
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
                init_kwargs[attr] = getattr(self, f'aggregate_{attr}')
            else:
                raise ValueError(f'{attr} has incompatible types across '
                                 f'instances!')

        super().__init__(**init_kwargs)

    def aggregate_fun_sensi_orders(self, x, sensi_orders):
        rvals = [
            objective.fun(x, sensi_orders)
            for objective in self.objectives
        ]
        return {
            key: sum(rval[key]) for rval in rvals
            for key in rvals[0]
        }

    def aggregate_res_sensi_orders(self, x, sensi_orders):
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

        # skip iobj=0 after initialization, stack matrices
        for iobj in range(1, len(self.objectives)):
            rval = self.objectives[iobj].res(x, sensi_orders)
            if res is not None:
                res = np.hstack([res, np.asarray(rval['res'])])
            if sres is not None:
                sres = np.vstack([sres, np.asarray(rval['sres'])])

        # transform results to dict
        result = dict()
        if res is not None:
            result['res'] = res
        if sres is not None:
            result['sres'] = sres

        return result

    def aggregate_res(self, x):
        res = self.objectives[0].res(x)
        for iobj in range(1, len(self.objectives)):
            res = np.hstack([res, np.asarray(self.objectives[iobj].res(x))])

        return res

    def aggregate_sres(self, x):
        sres = self.objectives[0].sres(x)
        for iobj in range(1, len(self.objectives)):
            sres = np.vstack([sres, np.asarray(self.objectives[iobj].sres(x))])

        return sres

    def aggregate_fun(self, x):
        return sum(objective.fun(x) for objective in self.objectives)

    def aggregate_grad(self, x):
        return sum(objective.grad(x) for objective in self.objectives)

    def aggregate_hess(self, x):
        return sum(objective.hess(x) for objective in self.objectives)

    def aggregate_hessp(self, x):
        return sum(objective.hessp(x) for objective in self.objectives)


def _check_boolean_value_consistent(objectives, attr):
    values = set(
        getattr(objective, attr)
        for objective in objectives
    )
    if len(values) > 1:
        raise ValueError(f'{attr} of all objectives must have a consistent '
                         f'value!')



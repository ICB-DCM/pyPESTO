"""
This adds an interface for the construction of loss functions
incorporating aesara models. This permits computation of derivatives using a
combination of objective based methods and aeara based backpropagation
"""

import numpy as np
import copy

from .base import ObjectiveBase, ResultDict
from .constants import MODE_FUN, FVAL, GRAD, HESS, RDATAS

from typing import Tuple, Optional, Sequence

try:
    import aesara
    import aesara.tensor as aet
    from aesara.tensor.var import TensorVariable
    from aesara.graph.op import Op
except ImportError:
    aesara = aet = Op = TensorVariable = None


class AesaraObjective(ObjectiveBase):
    """
    Wrapper around an ObjectiveBase which computes the gradient at each
    evaluation, caching it for later calls.
    Caching is only enabled after the first time the gradient is asked for
    and disabled whenever the cached gradient is not used,
    in order not to increase computation time for derivative-free samplers.

    Parameters
    ----------
    objective:
        The `pypesto.ObjectiveBase` to wrap.
    aet_x:
        Tensor variables that that define the variables of `aet_fun`
    aet_fun:
        Aesara function that maps `aet_x` to the variables of `objective`
    coeff:
        Multiplicative coefficient for objective
    """

    def __init__(self,
                 objective: ObjectiveBase,
                 aet_x: TensorVariable,
                 aet_fun: TensorVariable,
                 coeff: Optional[float] = 1.,
                 x_names: Sequence[str] = None):
        if not isinstance(objective, ObjectiveBase):
            raise TypeError('objective must be an ObjectiveBase instance')
        if not objective.check_mode(MODE_FUN):
            raise NotImplementedError(
                f'objective must support mode={MODE_FUN}')
        super().__init__(x_names)
        self.base_objective = objective

        self.aet_x = aet_x
        self.aet_fun = aet_fun
        self._coeff = coeff

        self.obj_op = AesaraObjectiveOp(self, self._coeff)

        # compiled function
        if objective.has_fun:
            self.afun = aesara.function(
                [aet_x], self.obj_op(aet_fun)
            )

        # compiled gradient
        if objective.has_grad:
            self.agrad = aesara.function(
                [aet_x], aesara.grad(self.obj_op(aet_fun), [aet_x])
            )

        # compiled hessian
        if objective.has_hess:
            self.ahess = aesara.function(
                [aet_x], aesara.gradient.hessian(self.obj_op(aet_fun), [aet_x])
            )

        # compiled input mapping
        self.infun = aesara.function(
            [aet_x], aet_fun
        )

        # temporary storage for evaluation results of objective
        self.inner_ret: ResultDict = {}

    def check_mode(self, mode) -> bool:
        return mode == MODE_FUN

    def check_sensi_orders(self, sensi_orders, mode) -> bool:
        if not self.check_mode(mode):
            return False
        else:
            return self.base_objective.check_sensi_orders(sensi_orders, mode)

    def call_unprocessed(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            **kwargs
    ) -> ResultDict:

        # hess computation in aesara requires grad
        if 2 in sensi_orders and 1 not in sensi_orders:
            sensi_orders = (1, *sensi_orders)

        # this computes all the results from the inner objective, rendering
        # them accessible to aesara compiled functions

        set_return_dict, return_dict = ('return_dict' in kwargs,
                                        kwargs.pop('return_dict', False))
        self.inner_ret = self.base_objective(
            self.infun(x), sensi_orders, mode, return_dict=True,  **kwargs
        )
        if set_return_dict:
            kwargs['return_dict'] = return_dict
        ret = {}
        if RDATAS in self.inner_ret:
            ret[RDATAS] = self.inner_ret[RDATAS]
        if 0 in sensi_orders:
            ret[FVAL] = float(self.afun(x))
        if 1 in sensi_orders:
            ret[GRAD] = self.agrad(x)[0]
        if 2 in sensi_orders:
            ret[HESS] = self.ahess(x)[0]

        return ret

    def __deepcopy__(self, memodict=None):
        other = AesaraObjective(
            copy.deepcopy(self.base_objective), self.aet_x, self.aet_fun,
            self._coeff
        )

        return other


class AesaraObjectiveOp(Op):
    """
    Aesara wrapper around a (non-normalized) log-probability function.

    Parameters
    ----------
    obj:
        Base aseara objective
    coeff:
        Multiplicative coefficient for the objective function value
    """

    itypes = [aet.dvector]  # expects a vector of parameter values when called
    otypes = [aet.dscalar]  # outputs a single scalar value (the log prob)

    def __init__(self,
                 obj: AesaraObjective,
                 coeff: Optional[float] = 1.):
        self._objective: AesaraObjective = obj
        self._coeff: float = coeff

        # initialize the sensitivity Op
        if obj.has_grad:
            self._log_prob_grad = AesaraObjectiveGradOp(obj, coeff)
        else:
            self._log_prob_grad = None

    def perform(self, node, inputs, outputs, params=None):
        # note that we use precomputed values from the outer
        # AesaraObjective.call_unprocessed here, which which means we can
        # ignore inputs here
        log_prob = self._coeff * self._objective.inner_ret[FVAL]
        outputs[0][0] = np.array(log_prob)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs
        log_prob_grad = self._log_prob_grad(theta)
        return [g[0] * log_prob_grad]


class AesaraObjectiveGradOp(Op):
    """
    Aesara wrapper around a (non-normalized) log-probability gradient function.
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.

    Parameters
    ----------
    obj:
        Base aseara objective
    coeff:
        Multiplicative coefficient for the objective function value
    """

    itypes = [aet.dvector]  # expects a vector of parameter values when called
    otypes = [aet.dvector]  # outputs a vector (the log prob grad)

    def __init__(self,
                 obj: AesaraObjective,
                 coeff: Optional[float] = 1.):
        self._objective: AesaraObjective = obj
        self._coeff: float = coeff

        if obj.has_hess:
            self._log_prob_hess = AesaraObjectiveHessOp(obj, coeff)
        else:
            self._log_prob_hess = None

    def perform(self, node, inputs, outputs, params=None):
        # note that we use precomputed values from the outer
        # AesaraObjective.call_unprocessed here, which which means we can
        # ignore inputs here
        log_prob_grad = self._coeff * self._objective.inner_ret[GRAD]
        outputs[0][0] = log_prob_grad

    def grad(self, inputs, g):
        # the method that calculates the hessian - it actually returns the
        # vector-hessian product - g[0] is a vector of parameter values
        theta, = inputs
        log_prob_hess = self._log_prob_hess(theta)
        return [g[0].dot(log_prob_hess)]


class AesaraObjectiveHessOp(Op):
    """
    Aesara wrapper around a (non-normalized) log-probability Hessian function.
    This Op will be called with a vector of values and also return a matrix of
    values - the Hessian in each dimension.

    Parameters
    ----------
    obj:
        Base aseara objective
    coeff:
        Multiplicative coefficient for the objective function value
    """

    itypes = [aet.dvector]
    otypes = [aet.dmatrix]

    def __init__(self, obj:
                 AesaraObjective,
                 coeff: Optional[float] = 1.):
        self._objective: AesaraObjective = obj
        self._coeff: float = coeff

    def perform(self, node, inputs, outputs, params=None):
        # note that we use precomputed values from the outer
        # AesaraObjective.call_unprocessed here, which which means we can
        # ignore inputs here
        log_prob_hess = self._coeff * self._objective.inner_ret[HESS]
        outputs[0][0] = log_prob_hess

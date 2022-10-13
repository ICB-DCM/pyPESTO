"""Enhanced Scatter Search.

See papers on ESS [EgeaBal2009]_, CESS [VillaverdeEge2012]_ and
saCeSS [PenasGon2017]_.

References
==========

.. [EgeaBal2009] 'Dynamic Optimization of Nonlinear Processes with an Enhanced
   Scatter Search Method', Jose A. Egea, Eva Balsa-Canto,
   María-Sonia G. García, and Julio R. Banga, Ind. Eng. Chem. Res.
   2009, 48, 9, 4388–4401. https://doi.org/10.1021/ie801717t

.. [EgeaMar2010] 'An evolutionary method for complex-process optimization',
   Jose A. Egea, Rafael Martí, Julio R. Banga, Computers & Operations Research,
   2010, 37, 2, 315-324. https://doi.org/10.1016/j.cor.2009.05.003

.. [VillaverdeEge2012] 'A cooperative strategy for parameter estimation in
   large scale systems biology models', Villaverde, A.F., Egea, J.A. & Banga,
   J.R. BMC Syst Biol 2012, 6, 75. https://doi.org/10.1186/1752-0509-6-75

.. [PenasGon2017] 'Parameter estimation in large-scale systems biology models:
   a parallel and self-adaptive cooperative strategy', David R. Penas,
   Patricia González, Jose A. Egea, Ramón Doallo and Julio R. Banga,
   BMC Bioinformatics 2017, 18, 52. https://doi.org/10.1186/s12859-016-1452-4
"""
import enum
import logging
import time
from typing import Tuple

import numpy as np

import pypesto.optimize
from pypesto import OptimizerResult, Problem
from pypesto.startpoint import StartpointMethod

from .function_evaluator import FunctionEvaluator
from .refset import RefSet

logger = logging.getLogger(__name__)

__all__ = ['ESSOptimizer', 'ESSExitFlag']


class ESSExitFlag(int, enum.Enum):
    """Exit flags used by :class:`ESSOptimizer`."""

    # ESS did not run/finish yet
    DID_NOT_RUN = 0
    # Exited after reaching maximum number of iterations
    MAX_ITER = -1
    # Exited after exhausting function evaluation budget
    MAX_EVAL = -2


class ESSOptimizer:
    """Enhanced Scatter Search (ESS) global optimization.

    .. note: Does not implement any constraint handling yet

    For plausible values of hyperparameters, see VillaverdeEge2012.

    Parameters
    ----------
    dim_refset:
        Size of the ReferenceSet
    max_iter:
        Maximum number of ESS iterations.
    local_n1:
        Minimum number of function evaluations before first local search.
    local_n2:
        Minimum number of function evaluations between consecutive local
        searches.
    local_optimizer:
        Local optimizer for refinement, or ``None`` to skip local searches.
    n_diverse:
        Number of samples to choose from to construct the initial RefSet
    max_eval:
        Maximum number of objective functions allowed. This criterion is
        only checked once per iteration, not after every objective evaluation,
        so the actual number of function evaluations may exceed this value.
    """

    def __init__(
        self,
        *,
        max_iter: int,
        local_n1: int,
        local_n2: int,
        dim_refset: int,
        local_optimizer: 'pypesto.optimize.Optimizer' = None,
        max_eval=np.inf,
        n_diverse: int = None,
        n_threads=1,
    ):
        # Hyperparameters
        self.local_n1 = local_n1
        self.local_n2 = local_n2
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.dim_refset = dim_refset
        self.local_optimizer = local_optimizer
        self.n_diverse = n_diverse
        self.n_threads = n_threads
        # quality vs diversity balancing factor [0, 1];
        #  0 = only quality; 1 = only diversity
        self.balance = 0.5
        # After how many iterations a stagnated solution is to be replaced by
        #  a random one. Default value taken from [EgeaMar2010]_
        self.n_change = 20
        # Only perform local search from best solution
        self.local_only_best_sol = False

        self._initialize()

    def _initialize(self):
        # RefSet
        self.refset = None
        # Overall best parameters found so far
        self.x_best = None
        # Overall best function value found so far
        self.fx_best = np.inf
        # Final parameters from local searches
        self.local_solutions = []
        # Index of current iteration
        self.n_iter = 0
        # Number of function evaluations at which the last local search took
        #  place
        self.last_local_search_neval = 0
        # Whether self.x_best has changed in the current iteration
        self.x_best_has_changed = False
        self.exit_flag = ESSExitFlag.DID_NOT_RUN
        self.evaluator = None
        self.starttime = None

    def minimize(
        self, problem: Problem, startpoint_method: StartpointMethod
    ) -> pypesto.Result:
        """Minimize."""
        if self.n_diverse is None:
            # [EgeaMar2010]_ 2.1
            self.n_diverse = 10 * problem.dim

        self._initialize()
        self.evaluator = FunctionEvaluator(
            problem=problem,
            startpoint_method=startpoint_method,
            n_threads=self.n_threads,
        )
        self.x_best = np.full(
            shape=(self.evaluator.problem.dim,), fill_value=np.nan
        )
        self.refset = RefSet(dim=self.dim_refset, evaluator=self.evaluator)
        refset = self.refset
        self.starttime = time.time()

        # Initial RefSet generation
        self.refset.initialize(n_diverse=self.n_diverse)

        # [PenasGon2017]_ Algorithm 1
        while self._keep_going():
            self.x_best_has_changed = False

            refset.sort()
            self._report_iteration()
            refset.prune_too_close()

            # Combination method
            x_best_children, fx_best_children = self._combine_solutions()

            # Go-beyond strategy
            self._go_beyond(x_best_children, fx_best_children)

            # Local search
            if self.local_optimizer is not None:
                self._do_local_search(x_best_children, fx_best_children)

            # replace refset members by best children where an improvement
            #  was made. replace stuck members by random points.
            for i in range(refset.dim):
                if fx_best_children[i] < refset.fx[i]:
                    refset.update(i, x_best_children[i], fx_best_children[i])
                else:
                    refset.n_stuck[i] += 1
                    if refset.n_stuck[i] > self.n_change:
                        refset.replace_by_random(i)

            self.n_iter += 1

        self._report_final()
        return self._create_result()

    def _create_result(self) -> pypesto.Result:
        """Create the result object.

        Currently, this returns the overall best value and the final RefSet.
        """
        common_result_fields = {
            'exitflag': self.exit_flag,
            # meaningful? this is the overall time, and identical for all
            #  reported points
            'time': time.time() - self.starttime,
            'n_fval': self.evaluator.n_eval,
        }
        i_result = 0
        result = pypesto.Result(problem=self.evaluator.problem)

        # save global best
        optimizer_result = pypesto.OptimizerResult(
            id=str(i_result),
            x=self.x_best,
            fval=self.fx_best,
            message="Global best",
            **common_result_fields,
        )
        # TODO DW: Create a single History with the global best?
        result.optimize_result.append(optimizer_result)

        # save refset
        for i in range(self.refset.dim):
            i_result += 1
            result.optimize_result.append(
                pypesto.OptimizerResult(
                    id=str(i_result),
                    x=self.refset.x[i],
                    fval=self.refset.fx[i],
                    message=f"RefSet[{i}]",
                    **common_result_fields,
                )
            )

        # TODO DW: also save local solutions?
        #  (need to track fvals or re-evaluate)

        return result

    def _keep_going(self) -> bool:
        """Check exit criteria.

        Returns
        -------
        ``True`` if not of the exit criteria is met, ``False`` otherwise.
        """
        # TODO DW which further stopping criteria: gtol, fatol, frtol?

        if self.n_iter >= self.max_iter:
            self.exit_flag = ESSExitFlag.MAX_ITER
            return False

        if self.evaluator.n_eval >= self.max_eval:
            self.exit_flag = ESSExitFlag.MAX_EVAL
            return False

        return True

    def _combine_solutions(self) -> Tuple[np.array, np.array]:
        """Combine solutions and evaluate."""
        y = np.zeros(shape=(self.refset.dim, self.evaluator.problem.dim))
        fy = np.full(shape=self.refset.dim, fill_value=np.inf)
        for i in range(self.refset.dim):
            xs_new = np.vstack(
                tuple(
                    self._combine(i, j)
                    for j in range(self.refset.dim)
                    if i != j
                ),
            )
            fxs_new = self.evaluator.multiple(xs_new)
            best_idx = np.argmin(fxs_new)
            fy[i] = fxs_new[best_idx]
            y[i] = xs_new[best_idx]
        return y, fy

    def _combine(self, i, j) -> np.array:
        # combine solutions
        # see [EgeaBal2009]_ Section 3.2
        # TODO DW: will that always yield admissible points?
        if i == j:
            raise ValueError("i == j")
        x = self.refset.x

        d = x[j] - x[i]
        alpha = np.sign(j - i)
        beta = (np.abs(j - i) - 1) / (self.refset.dim - 2)
        c1 = x[i] - d * (1 + alpha * beta)
        c2 = x[i] - d * (1 - alpha * beta)
        r = np.random.uniform(0, 1, self.evaluator.problem.dim)
        return c1 + (c2 - c1) * r

    def _do_local_search(self, x_best_children, fx_best_children):
        """
        Perform a local search to refine the next generation.

        See [PenasGon2017]_ Algorithm 2.
        """
        if self.local_only_best_sol and self.x_best_has_changed:
            logger.debug("Local search only from best point.")
            local_search_x0 = self.x_best
            local_search_fx0 = self.fx_best
        # first local search?
        elif (
            not self.local_solutions and self.evaluator.n_eval >= self.local_n1
        ):
            logger.debug(
                "First local search from best point due to "
                f"local_n1={self.local_n1}."
            )
            local_search_x0 = self.x_best
            local_search_fx0 = self.fx_best
        elif (
            self.local_solutions
            and self.evaluator.n_eval - self.last_local_search_neval
            >= self.local_n2
        ):
            quality_order = np.argsort(fx_best_children)
            # compute minimal distance between the best children and all local
            #  optima found so far
            min_distances = np.array(
                np.min(
                    np.linalg.norm(y_i - local_solution)
                    for local_solution in self.local_solutions
                )
                for y_i in x_best_children
            )
            # sort by furthest distance to existing local optima
            diversity_order = np.argsort(min_distances)[::-1]
            # balance quality and diversity (small score is better)
            priority = (
                1 - self.balance
            ) * quality_order + self.balance * diversity_order
            chosen_child_idx = np.argmin(priority)
            local_search_x0 = x_best_children[chosen_child_idx]
            local_search_fx0 = fx_best_children[chosen_child_idx]
        else:
            return

        # actual local search
        # TODO DW: try alternatives if it fails on initial point?
        optimizer_result: OptimizerResult = self.local_optimizer.minimize(
            problem=self.evaluator.problem,
            x0=local_search_x0,
            id="0",
        )
        # add function evaluations during local search to our function
        #  evaluation counter (NOTE: depending on the setup, we might neglect
        #  gradient evaluations).
        self.evaluator.n_eval += optimizer_result.n_fval
        self.evaluator.n_eval_round += optimizer_result.n_fval

        logger.debug(
            f"Local search: {local_search_fx0} -> " f"{optimizer_result.fval}"
        )
        self.local_solutions.append(optimizer_result.x)

        self._maybe_update_global_best(
            optimizer_result.x, optimizer_result.fval
        )
        self.last_local_search_neval = self.n_iter
        self.evaluator.reset_round_counter()

    def _maybe_update_global_best(self, x, fx):
        if fx < self.fx_best:
            self.x_best = x[:]
            self.fx_best = fx
            self.x_best_has_changed = True

    def _go_beyond(self, x_best_children, fx_best_children):
        """Apply go-beyond strategy.

        See [Egea2009]_ algorithm 1 + section 3.4
        """
        for i in range(self.refset.dim):
            if fx_best_children[i] >= self.refset.fx[i]:
                continue

            # offspring is better than parent
            x_parent = self.refset.x[i]
            fx_parent = self.refset.fx[i]
            x_child = x_best_children[i]
            fx_child = fx_best_children[i]
            improvement = 1
            # Multiplier used in determining the hyper-rectangle from which to
            # sample children. Will be increased in case of 2 consecutive
            # improvements.
            # (corresponds to 1/\Lambda in [Egea2009]_ algorithm 1)
            go_beyound_factor = 1
            while fx_child < fx_parent:
                # update best child
                x_best_children[i] = x_child
                fx_best_children[i] = fx_child

                # create new solution, child becomes parent
                x_new = np.random.uniform(
                    low=x_child - (x_parent - x_child) * go_beyound_factor,
                    high=x_child,
                )
                x_parent = x_child
                fx_parent = fx_child
                x_child = x_new
                fx_child = self.evaluator.single(x_child)

                improvement += 1
                if improvement == 2:
                    go_beyound_factor *= 2
                    improvement = 0

            # update overall best?
            self._maybe_update_global_best(
                x_best_children[i], fx_best_children[i]
            )

    def _report_iteration(self):
        if self.n_iter == 0:
            logger.info("iter | best | nf | refset         |")

        with np.printoptions(
            edgeitems=30,
            linewidth=100000,
            formatter={"float": lambda x: "%.3g" % x},
        ):
            logger.info(
                f"{self.n_iter:4} | {self.fx_best:+.2E} | "
                f"{self.evaluator.n_eval} "
                f"| {self.refset.fx} | {len(self.local_solutions)}"
            )

    def _report_final(self):
        with np.printoptions(
            edgeitems=30,
            linewidth=100000,
            formatter={"float": lambda x: "%.3g" % x},
        ):
            logger.info(
                f"-- Stopping after {self.n_iter} iterations. "
                f"Final refset: {np.sort(self.refset.fx)} "
                f"num local solutions: {len(self.local_solutions)}"
            )

        logger.info(f"Best fval {self.fx_best}")

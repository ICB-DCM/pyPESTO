"""Enhanced Scatter Search.

See papers on ESS [EgeaBal2009]_ and saCeSS [PenasGon2017]_.

References
==========

.. [EgeaBal2009] 'Dynamic Optimization of Nonlinear Processes with an Enhanced
  Scatter Search Method', Jose A. Egea, Eva Balsa-Canto, María-Sonia G. García,
  and Julio R. Banga, Ind. Eng. Chem. Res. 2009, 48, 9, 4388–4401.
  https://doi.org/10.1021/ie801717t
.. [PenasGon2017] 'Parameter estimation in large-scale systems biology models:
   a parallel and self-adaptive cooperative strategy', David R. Penas,
   Patricia González, Jose A. Egea, Ramón Doallo and Julio R. Banga,
   BMC Bioinformatics 2017, 18, 52 https://doi.org/10.1186/s12859-016-1452-4
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Sequence, Tuple

import numpy as np

import pypesto.optimize
from pypesto import Problem
from pypesto.startpoint import StartpointMethod

logger = logging.getLogger(__name__)

__all__ = ['OptimizerESS']


class FunctionEvaluator:
    """Wrapper for optimization problem and startpoint method.

    Takes care of (not yet parallel) function evaluations, startpoint
    sampling, and tracks number of function evaluations.

    Parameters
    ----------
    problem: The problem
    startpoint_method: Method for choosing feasible parameters
    """

    def __init__(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod,
        n_threads: int,
    ):
        self.problem = problem
        self.startpoint_method = startpoint_method
        self.n_eval = 0
        self.n_eval_round = 0
        self.n_threads = n_threads
        self._thread_local = threading.local()
        self.executor = ThreadPoolExecutor(
            max_workers=self.n_threads,
            thread_name_prefix=__name__,
            initializer=self._initialize_worker,
            initargs=(self._thread_local,),
        )

    def single(self, x: np.array) -> float:
        self.n_eval += 1
        self.n_eval_round += 1
        return self.problem.objective(x)

    def multiple(self, xs: Sequence):
        res = np.fromiter(
            # map(self.single, xs),
            self.executor.map(
                self._evalute_on_worker, ((self._thread_local, x) for x in xs)
            ),
            dtype=float,
        )
        self.n_eval += len(xs)
        self.n_eval_round += len(xs)
        return res

    def single_random(self) -> Tuple[np.array, float]:
        x = fx = np.nan
        while not np.isfinite(fx):
            x = self.startpoint_method(n_starts=1, problem=self.problem)[0]
            fx = self.single(x)
        return x, fx

    def multiple_random(self, n: int):
        # TODO: check for finite fval (gradient does not matter)
        xs = self.startpoint_method(n_starts=n, problem=self.problem)
        return xs, self.multiple(xs)

    def reset_counter(self):
        self.n_eval = 0
        self.reset_round_counter()

    def reset_round_counter(self):
        self.n_eval_round = 0

    def _initialize_worker(self, local):
        local.objective = deepcopy(self.problem.objective)

    @staticmethod
    def _evalute_on_worker(local_and_x):
        local, x = local_and_x
        return local.objective(x)


class RefSet:
    """Scatter search reference set.

    Parameters
    ----------
    dim: Reference set size
    evaluator: Function evaluator
    """

    def __init__(self, dim: int, evaluator: FunctionEvaluator):
        self.dim = dim
        self.evaluator = evaluator
        # epsilon in [PenasGon2017]_
        self.proximity_threshold = 1e-3

        self.fx = np.full(shape=(dim,), fill_value=np.inf)
        self.x = np.full(
            shape=(dim, self.evaluator.problem.dim), fill_value=np.nan
        )
        self.n_stuck = np.zeros(shape=[dim])

    def sort(self):
        """Sort RefSet by quality."""
        order = np.argsort(self.fx)
        self.fx = self.fx[order]
        self.x = self.x[order]
        self.n_stuck = self.n_stuck[order]

    def initialize(self, n_diverse: int):
        """Create initial reference set with random parameters.

        Generate initial RefSet with 0.5*`dim_refset` best solutions from
        `n_diverse` random points. Fill the rest with random points.
        """
        # sample n_diverse points
        xtmp, ftmp = self.evaluator.multiple_random(n_diverse)

        # create initial refset with 50% best values
        order = np.argsort(ftmp)
        num_best = int(self.dim / 2)
        self.x[:num_best] = xtmp[order[:num_best]]
        self.fx[:num_best] = ftmp[order[:num_best]]

        # ... and 50% others
        random_idxs = np.random.choice(
            order[num_best:], size=self.dim - num_best, replace=False
        )
        self.x[num_best:] = xtmp[random_idxs]
        self.fx[num_best:] = ftmp[random_idxs]

    def prune_too_close(self):
        # prune too close ones, pair-wise comparison
        x = self.x
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # check proximity
                while (
                    np.max(np.abs((x[i] - x[j]) / x[j]))
                    <= self.proximity_threshold
                ):
                    # too close. replace x_j.
                    x[j], self.fx[j] = self.evaluator.single_random()
                    self.sort()

    def update(self, i, x, fx):
        """Update a entry RefSet entry."""
        self.x[i] = x
        self.fx[i] = fx
        self.n_stuck[i] = 0

    def replace_by_random(self, i: int):
        self.x[i], self.fx[i] = self.evaluator.single_random()
        self.n_stuck[i] = 0


class OptimizerESS:
    """Enhanced Scatter Search (ESS) global optimization.

    .. note: Does not implement any constraint handling yet

    Parameters
    ----------
    dim_refset: Size of the ReferenceSet
    max_iter: Maximum number of ESS iterations.
    local_n1: Minimum number of function evaluations before first local search.
    local_n2: Minimum number of function evaluations between consecutive local
        searches.
    local_optimizer: Local optimizer for refinement, or ``None`` to skip local
        searches.
    n_diverse: Number of samples to choose from to construct the initial RefSet
    """

    def __init__(
        self,
        *,
        max_iter: int,
        local_n1: int,
        local_n2: int,
        dim_refset: int,
        local_optimizer: pypesto.optimize.Optimizer = None,
        n_diverse: int = None,
        n_threads=1,
    ):
        # Hyperparameters
        # TODO number of iterations or function evaluations? paper ambiguous
        self.local_n1 = local_n1
        self.local_n2 = local_n2
        self.max_iter = max_iter
        self.dim_refset = dim_refset
        self.local_optimizer = local_optimizer
        self.n_diverse = n_diverse or 10 * dim_refset
        self.n_threads = n_threads
        # quality vs diversity balancing factor [0, 1];
        #  0 = only quality; 1 = only diversity
        self.balance = 0.5
        # after how many iterations a stagnated solution is to be replaced by
        #  a random one
        self.n_change = 5  # TODO
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
        # Iteration in which the last local search took place
        self.last_local_search_iteration = 0
        # Whether self.x_best has changed in the current iteration
        self.x_best_has_changed = False

        self.evaluator = None
        self.starttime = None

    def minimize(self, problem: Problem, startpoint_method: StartpointMethod):
        """Minimize."""
        logger.setLevel(logging.DEBUG)
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

    def _create_result(self):
        # TODO what to return here:
        #  only the single best value?
        #  the local solutions + the best
        #  the local solutions + refset + best?
        duration = time.time() - self.starttime
        i_result = 0
        result = pypesto.Result(problem=self.evaluator.problem)
        # save global best
        optimizer_result = pypesto.OptimizerResult(
            id=str(i_result),
            x=self.x_best,
            fval=self.fx_best,
            # TODO
            message="Global best",
            # TODO
            exitflag=0,
            time=duration,
        )
        result.optimize_result.append(optimizer_result)

        # save refset

        for i in range(self.refset.dim):
            i_result += 1
            result.optimize_result.append(
                pypesto.OptimizerResult(
                    id=str(i_result),
                    x=self.refset.x[i],
                    fval=self.refset.fx[i],
                    # TODO
                    message=f"RefSet[{i}]",
                    # TODO
                    exitflag=0,
                    time=duration,
                )
            )
        # TODO save local solutions (need to track fvals or re-evaluate)

        return result

    def _keep_going(self):
        """Check exit criteria."""
        # TODO further stopping criteria: gtol, fatol, frtol

        if self.n_iter >= self.max_iter:
            return False

        return True

    def _combine_solutions(self) -> Tuple[np.array, np.array]:
        """Combine solutions and evaluate."""
        # TODO: move to refset
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

    def _combine(self, i, j):
        # combine solutions
        # see [EgeaBal2009]_ Section 3.2
        # TODO: will that always yield admissible points?
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
        elif not self.local_solutions and self.n_iter >= self.local_n1:
            logger.debug(
                f"First local search from best point due to local_n1={self.local_n1}."
            )
            local_search_x0 = self.x_best
            local_search_fx0 = self.fx_best
        elif (
            self.local_solutions
            and self.n_iter - self.last_local_search_iteration >= self.local_n2
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
        # TODO try alternatives if it fails on initial point?
        optimizer_result = self.local_optimizer.minimize(
            problem=self.evaluator.problem,
            x0=local_search_x0,
            id="0",  # TODO
        )
        logger.debug(
            f"Local search: {local_search_fx0} -> " f"{optimizer_result.fval}"
        )
        self.local_solutions.append(optimizer_result.x)

        self._maybe_update_global_best(
            optimizer_result.x, optimizer_result.fval
        )
        self.last_local_search_iteration = self.n_iter
        self.evaluator.reset_round_counter()

    def _maybe_update_global_best(self, x, fx):
        if fx < self.fx_best:
            self.x_best = x[:]
            self.fx_best = fx
            self.x_best_has_changed = True

    def _go_beyond(self, x_best_children, fx_best_children):
        """Apply go-beyond strategy.

        See [Egea2009]_ algorithm 1
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
            Lambda = 1
            while fx_child < fx_parent:
                # update best child
                x_best_children[i] = x_child
                fx_best_children[i] = fx_child

                # create new solution, child becomes parent
                x_new = np.random.uniform(
                    low=x_child - (x_parent - x_child) / Lambda,
                    high=x_child,
                )
                x_parent = x_child
                fx_parent = fx_child
                x_child = x_new
                fx_child = self.evaluator.single(x_child)

                improvement += 1
                if improvement == 2:
                    Lambda /= 2
                    improvement = 0

            # update overall best?
            self._maybe_update_global_best(
                x_best_children[i], fx_best_children[i]
            )

    def _report_iteration(self):
        logger.info(
            f"{self.n_iter:4} | {self.fx_best:+.2E} | {self.refset.fx}"
            f" | {len(self.local_solutions)}"
        )

    def _report_final(self):
        logger.info(
            f"-- Stopping after {self.n_iter} iterations. "
            f"Final refset: {np.sort(self.refset.fx)} "
            f"num local solutions: {len(self.local_solutions)}"
        )

        logger.info(f"Best fval {self.fx_best}")

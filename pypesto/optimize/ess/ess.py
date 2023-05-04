"""Enhanced Scatter Search.

See papers on ESS [EgeaBal2009]_ [EgeaMar2010]_, CESS [VillaverdeEge2012]_ and
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
from typing import List, Optional, Tuple

import numpy as np

import pypesto.optimize
from pypesto import OptimizerResult, Problem
from pypesto.startpoint import StartpointMethod

from .function_evaluator import (
    FunctionEvaluator,
    FunctionEvaluatorMP,
    FunctionEvaluatorMT,
)
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
    # Exited after exhausting wall-time budget
    MAX_TIME = -3


class ESSOptimizer:
    """Enhanced Scatter Search (ESS) global optimization.

    .. note: Does not implement any constraint handling beyond box constraints
    """

    def __init__(
        self,
        *,
        max_iter: int = 10**100,
        dim_refset: int = None,
        local_n1: int = 1,
        local_n2: int = 10,
        balance: float = 0.5,
        local_optimizer: 'pypesto.optimize.Optimizer' = None,
        max_eval=np.inf,
        n_diverse: int = None,
        n_procs=None,
        n_threads=None,
        max_walltime_s=None,
    ):
        """Construct new ESS instance.

        For plausible values of hyperparameters, see VillaverdeEge2012.

        Parameters
        ----------
        dim_refset:
            Size of the ReferenceSet. Note that in every iteration at least
            ``dim_refset**2 - dim_refset`` function evaluations will occur.
        max_iter:
            Maximum number of ESS iterations.
        local_n1:
            Minimum number of iterations before first local search.
        local_n2:
            Minimum number of iterations between consecutive local
            searches. Maximally one local search per performed in each
            iteration.
        local_optimizer:
            Local optimizer for refinement, or ``None`` to skip local searches.
        n_diverse:
            Number of samples to choose from to construct the initial RefSet
        max_eval:
            Maximum number of objective functions allowed. This criterion is
            only checked once per iteration, not after every objective
            evaluation, so the actual number of function evaluations may exceed
            this value.
        max_walltime_s:
            Maximum walltime in seconds. Will only be checked between local
            optimizations and other simulations, and thus, may be exceeded by
            the duration of a local search.
        balance:
            Quality vs diversity balancing factor [0, 1];
            0 = only quality; 1 = only diversity
        n_procs:
            Number of parallel processes to use for parallel function
            evaluation. Mutually exclusive with `n_threads`.
        n_threads:
            Number of parallel threads to use for parallel function evaluation.
            Mutually exclusive with `n_procs`.
        """
        # Hyperparameters
        self.local_n1: int = local_n1
        self.local_n2: int = local_n2
        self.max_iter: int = max_iter
        self.max_eval: int = max_eval
        self.dim_refset: int = dim_refset
        self.local_optimizer: Optional[
            'pypesto.optimize.Optimizer'
        ] = local_optimizer
        self.n_diverse: int = n_diverse
        if n_procs is not None and n_threads is not None:
            raise ValueError(
                "`n_procs` and `n_threads` are mutually exclusive."
            )
        self.n_procs: Optional[int] = n_procs
        self.n_threads: Optional[int] = n_threads
        self.balance: float = balance
        # After how many iterations a stagnated solution is to be replaced by
        #  a random one. Default value taken from [EgeaMar2010]_
        self.n_change: int = 20
        # Only perform local search from best solution
        self.local_only_best_sol: bool = False
        self.max_walltime_s = max_walltime_s
        self._initialize()
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}-{id(self)}"
        )

    def _initialize(self):
        """(Re-)Initialize."""
        # RefSet
        self.refset: Optional[RefSet] = None
        # Overall best parameters found so far
        self.x_best: Optional[np.array] = None
        # Overall best function value found so far
        self.fx_best: float = np.inf
        # Final parameters from local searches
        self.local_solutions: List[np.array] = []
        # Index of current iteration
        self.n_iter: int = 0
        # ESS iteration at which the last local search took place
        self.last_local_search_niter: int = 0
        # Whether self.x_best has changed in the current iteration
        self.x_best_has_changed: bool = False
        self.exit_flag: ESSExitFlag = ESSExitFlag.DID_NOT_RUN
        self.evaluator: Optional[FunctionEvaluator] = None
        self.starttime: Optional[float] = None

    def minimize(
        self,
        problem: Problem = None,
        startpoint_method: StartpointMethod = None,
        refset: Optional[RefSet] = None,
    ) -> pypesto.Result:
        """Minimize the given objective.

        Parameters
        ----------
        problem:
            Problem to run ESS on.
        startpoint_method:
            Method for choosing starting points.
        refset:
            The initial RefSet or ``None`` to auto-generate.
        """
        self._initialize()
        self.starttime = time.time()

        if (
            refset is None and (problem is None or startpoint_method is None)
        ) or (
            refset is not None
            and (problem is not None or startpoint_method is not None)
        ):
            raise ValueError(
                "Either `refset` or `problem` and `startpoint_method` "
                "has to be provided."
            )
        # generate initial RefSet if not provided
        if refset is None:
            if self.dim_refset is None:
                raise ValueError(
                    "Either refset or dim_refset have to be provided."
                )
            # [EgeaMar2010]_ 2.1
            self.n_diverse = self.n_diverse or 10 * problem.dim
            if self.n_procs:
                self.evaluator = FunctionEvaluatorMP(
                    problem=problem,
                    startpoint_method=startpoint_method,
                    n_procs=self.n_procs,
                )
            else:
                self.evaluator = FunctionEvaluatorMT(
                    problem=problem,
                    startpoint_method=startpoint_method,
                    n_threads=self.n_threads or 1,
                )

            self.refset = RefSet(dim=self.dim_refset, evaluator=self.evaluator)
            # Initial RefSet generation
            self.refset.initialize_random(n_diverse=self.n_diverse)
            refset = self.refset
        else:
            self.refset = refset
            problem = refset.evaluator.problem

        self.evaluator = refset.evaluator
        self.x_best = np.full(
            shape=(self.evaluator.problem.dim,), fill_value=np.nan
        )
        # initialize global best from initial refset
        for x, fx in zip(refset.x, refset.fx):
            self._maybe_update_global_best(x, fx)

        # [PenasGon2017]_ Algorithm 1
        while self._keep_going():
            self.x_best_has_changed = False

            refset.sort()
            self._report_iteration()
            refset.prune_too_close()

            # Apply combination method to update the RefSet
            x_best_children, fx_best_children = self._combine_solutions()

            # Go-beyond strategy to further improve the new combinations
            self._go_beyond(x_best_children, fx_best_children)

            # Maybe perform a local search
            if self.local_optimizer is not None:
                self._do_local_search(x_best_children, fx_best_children)

            # Replace RefSet members by best children where an improvement
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
            'optimizer': str(self),
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

        if (
            self.max_walltime_s is not None
            and time.time() - self.starttime > self.max_walltime_s
        ):
            self.exit_flag = ESSExitFlag.MAX_TIME
            return False

        return True

    def _combine_solutions(self) -> Tuple[np.array, np.array]:
        """Combine solutions and evaluate.

        Creates the next generation from the RefSet by pair-wise combinations
        of all RefSet members. Creates ``RefSet.dim ** 2 - RefSet.dim`` new
        parameter vectors, tests them, and keeps the best child of each parent.

        Returns
        -------
        y:
            Contains the best of all pairwise combinations of all RefSet
            members, for each RefSet members.
        fy:
            The corresponding objective values.
        """
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
        """Combine RefSet members ``i`` and ``j``.

        Samples a new point from a biased hyper-rectangle derived from the
        given parents, favoring the direction of the better parent.

        Assumes that the RefSet is sorted by quality.

        See [EgeaBal2009]_ Section 3.2 for details.

        Parameters
        ----------
        i:
            Index of first RefSet member for recombination
        j:
            Index of second RefSet member for recombination

        Returns
        -------
        A new parameter vector.
        """
        if i == j:
            raise ValueError("i == j")
        x = self.refset.x

        d = x[j] - x[i]
        alpha = np.sign(j - i)
        beta = (np.abs(j - i) - 1) / (self.refset.dim - 2)
        c1 = x[i] - d * (1 + alpha * beta)
        c2 = x[i] - d * (1 - alpha * beta)

        # this will not always yield admissible points -> clip to bounds
        ub, lb = self.evaluator.problem.ub, self.evaluator.problem.lb
        c1 = np.fmax(np.fmin(c1, ub), lb)
        c2 = np.fmax(np.fmin(c2, ub), lb)

        return np.random.uniform(
            low=c1, high=c2, size=self.evaluator.problem.dim
        )

    def _do_local_search(
        self, x_best_children: np.array, fx_best_children: np.array
    ) -> None:
        """
        Perform a local search to refine the next generation.

        See [PenasGon2017]_ Algorithm 2.
        """
        if self.local_only_best_sol and self.x_best_has_changed:
            self.logger.debug("Local search only from best point.")
            local_search_x0 = self.x_best
            local_search_fx0 = self.fx_best
        # first local search?
        elif not self.local_solutions and self.n_iter >= self.local_n1:
            self.logger.debug(
                "First local search from best point due to "
                f"local_n1={self.local_n1}."
            )
            local_search_x0 = self.x_best
            local_search_fx0 = self.fx_best
        elif (
            self.local_solutions
            and self.n_iter - self.last_local_search_niter >= self.local_n2
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
            # compute priority, balancing quality and diversity
            #  (smaller value = higher priority)
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

        self.logger.info(
            f"Local search: {local_search_fx0} -> {optimizer_result.fval} "
            f" took {optimizer_result.time:.3g}s, finished with "
            f"{optimizer_result.exitflag}: {optimizer_result.message}"
        )
        self.local_solutions.append(optimizer_result.x)

        self._maybe_update_global_best(
            optimizer_result.x, optimizer_result.fval
        )
        self.last_local_search_niter = self.n_iter
        self.evaluator.reset_round_counter()

    def _maybe_update_global_best(self, x, fx):
        """Update the global best value if the provided value is better."""
        if fx < self.fx_best:
            self.x_best = x[:]
            self.fx_best = fx
            self.x_best_has_changed = True

    def _go_beyond(self, x_best_children, fx_best_children):
        """Apply go-beyond strategy.

        If a child is better than its parent, intensify search in that
        direction until no further improvement is made.

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
            go_beyond_factor = 1
            while fx_child < fx_parent:
                # update best child
                x_best_children[i] = x_child
                fx_best_children[i] = fx_child

                # create new solution, child becomes parent
                # hyper-rectangle for sampling child
                box_lb = x_child - (x_parent - x_child) * go_beyond_factor
                box_ub = x_child
                # clip to bounds
                ub, lb = self.evaluator.problem.ub, self.evaluator.problem.lb
                box_lb = np.fmax(np.fmin(box_lb, ub), lb)
                box_ub = np.fmax(np.fmin(box_ub, ub), lb)
                # sample parameters
                x_new = np.random.uniform(low=box_lb, high=box_ub)
                x_parent = x_child
                fx_parent = fx_child
                x_child = x_new
                fx_child = self.evaluator.single(x_child)

                improvement += 1
                if improvement == 2:
                    go_beyond_factor *= 2
                    improvement = 0

            # update overall best?
            self._maybe_update_global_best(
                x_best_children[i], fx_best_children[i]
            )

    def _report_iteration(self):
        """Log the current iteration."""
        if self.n_iter == 0:
            self.logger.info("iter | best | nf | refset         |")

        with np.printoptions(
            edgeitems=5,
            threshold=8,
            linewidth=100000,
            formatter={"float": lambda x: "%.3g" % x},
        ):
            self.logger.info(
                f"{self.n_iter:4} | {self.fx_best:+.2E} | "
                f"{self.evaluator.n_eval} "
                f"| {self.refset.fx} | {len(self.local_solutions)}"
            )

    def _report_final(self):
        """Log scatter search summary."""
        with np.printoptions(
            edgeitems=5,
            threshold=10,
            linewidth=100000,
            formatter={"float": lambda x: "%.3g" % x},
        ):
            self.logger.info(
                f"-- Final ESS fval after {self.n_iter} "
                f"iterations: {self.fx_best}. "
                f"Exit flag: {self.exit_flag.name}. "
                f"Num local solutions: {len(self.local_solutions)}."
            )
            self.logger.debug(f"Final refset: {np.sort(self.refset.fx)} ")

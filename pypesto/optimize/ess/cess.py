"""Cooperative Enhanced Scatter Search."""
import logging
import multiprocessing
import os
import time
from typing import Dict, List, Optional

import numpy as np

import pypesto.optimize
from pypesto import Problem
from pypesto.startpoint import StartpointMethod

from .ess import ESSExitFlag, ESSOptimizer
from .function_evaluator import FunctionEvaluator
from .refset import RefSet

logger = logging.getLogger(__name__)


class CESSOptimizer:
    r"""
    Cooperative Enhanced Scatter Search Optimizer (CESS).

    A cooperative scatter search algorithm based on [VillaverdeEge2012]_.
    In short, multiple scatter search instances with different hyperparameters
    are running in different threads/processes, and exchange information.
    Some instances focus on diversification while others focus on
    intensification. Communication happens at fixed time intervals.

    Proposed hyperparameter values in [VillaverdeEge2012]_:

    * ``dim_refset``: ``[0.5 n_parameter, 20 n_parameters]``
    * ``local_n2``: ``[0, 100]``
    * ``balance``: ``[0, 0.5]``
    * ``n_diverse``: ``[5 n_par, 20 n_par]``
    * ``max_eval``: such that :math:`\tau = log10(max_eval / n_par)` is in
      [2.5, 3.5], with a recommended default value of 2.5.

    .. [VillaverdeEge2012] 'A cooperative strategy for parameter estimation in
       large scale systems biology models', Villaverde, A.F., Egea,
       J.A. & Banga, J.R. BMC Syst Biol 2012, 6, 75.
       https://doi.org/10.1186/1752-0509-6-75

    Attributes
    ----------
    ess_init_args:
        List of argument dictionaries passed to
        :func:`ESSOptimizer.__init__`. The length of this list is the
        number of parallel ESS processes.
        Resource limits such as ``max_eval`` apply to a single CESS
        iteration, not to the full search.
    max_iter:
        Maximum number of CESS iterations.
    max_walltime_s:
        Maximum walltime in seconds. Will only be checked between local
        optimizations and other simulations, and thus, may be exceeded by
        the duration of a local search. Defaults to no limit.
    fx_best:
        The best objective value seen so far.
    x_best:
        Parameter vector corresponding to ``fx_best``.
    starttime:
        Starting time of the most recent optimization.
    i_iter:
        Current iteration number.
    """

    def __init__(
        self,
        ess_init_args: List[Dict],
        max_iter: int,
        max_walltime_s: float = np.inf,
    ):
        """Construct.

        Parameters
        ----------
        ess_init_args:
            List of argument dictionaries passed to
            :func:`ESSOptimizer.__init__`. The length of this list is the
            number of parallel ESS processes.
            Resource limits such as ``max_eval`` apply to a single CESS
            iteration, not to the full search.
        max_iter:
            Maximum number of CESS iterations.
        max_walltime_s:
            Maximum walltime in seconds. Will only be checked between local
            optimizations and other simulations, and thus, may be exceeded by
            the duration of a local search. Defaults to no limit.
        """
        self.max_walltime_s = max_walltime_s
        self.ess_init_args = ess_init_args
        self.max_iter = max_iter

        self._initialize()

    def _initialize(self):
        """(Re-)initialize."""
        self.starttime = time.time()
        self.i_iter = 0
        # Overall best parameters found so far
        self.x_best: Optional[np.array] = None
        # Overall best function value found so far
        self.fx_best: float = np.inf

    def minimize(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod,
    ) -> pypesto.Result:
        """Minimize the given objective using CESS.

        Parameters
        ----------
        problem:
            Problem to run ESS on.
        startpoint_method:
            Method for choosing starting points.
        """
        self._initialize()

        evaluator = FunctionEvaluator(
            problem=problem,
            startpoint_method=startpoint_method,
        )

        refsets = [
            RefSet(evaluator=evaluator, dim=ess_init_args['dim_refset'])
            for ess_init_args in self.ess_init_args
        ]
        for refset, ess_init_args in zip(refsets, self.ess_init_args):
            refset.initialize_random(
                n_diverse=ess_init_args.get('n_diverse', 10 * problem.dim)
            )

        while True:
            logger.info("-" * 50)
            self._report_iteration()
            self.i_iter += 1

            # run scatter searches
            ess_optimizers = self._run_scatter_searches(
                refsets=refsets,
            )
            # collect refsets from the different ESS runs
            refsets = [result.refset for result in ess_optimizers]

            # update best values from ESS results
            for result in ess_optimizers:
                self._maybe_update_global_best(result.x_best, result.fx_best)

            if not self._keep_going(i_eval=evaluator.n_eval):
                break

            # create refsets for the next iteration
            self._update_refsets(refsets=refsets, evaluator=evaluator)

            # TODO merge results
        self._report_final()

        # TODO what should the result look like?
        return self._create_result(problem, refsets)

    def _report_iteration(self):
        """Log the current iteration."""
        if self.max_iter == 0:
            logger.info("iter | best |")

        with np.printoptions(
            edgeitems=30,
            linewidth=100000,
            formatter={"float": lambda x: "%.3g" % x},
        ):
            logger.info(f"{self.i_iter:4} | {self.fx_best:+.2E} | ")

    def _report_final(self):
        """Log scatter search summary."""
        with np.printoptions(
            edgeitems=30,
            linewidth=100000,
            formatter={"float": lambda x: "%.3g" % x},
        ):
            logger.info(
                f"CESS finished with {self.exit_flag!r} "
                f"after {self.i_iter} iterations, "
                f"{time.time() - self.starttime:.3g}s. "
                # f"Num local solutions: {len(self.local_solutions)}."
            )
            # logger.info(f"Final refset: {np.sort(self.refset.fx)} ")
        logger.info(f"Best fval {self.fx_best}")

    def _create_result(
        self, problem: pypesto.Problem, refsets: List[RefSet]
    ) -> pypesto.Result:
        """Create the result object.

        Currently, this returns the overall best value and the final RefSet.
        """
        common_result_fields = {
            'exitflag': self.exit_flag,
            # meaningful? this is the overall time, and identical for all
            #  reported points
            'time': time.time() - self.starttime,
            # TODO
            # 'n_fval': self.evaluator.n_eval,
            'optimizer': str(self),
        }
        i_result = 0
        result = pypesto.Result(problem=problem)

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

        # save refsets
        for i_refset, refset in enumerate(refsets):
            for i in range(refset.dim):
                i_result += 1
                result.optimize_result.append(
                    pypesto.OptimizerResult(
                        id=str(i_result),
                        x=refset.x[i],
                        fval=refset.fx[i],
                        message=f"RefSet[{i_refset}][{i}]",
                        **common_result_fields,
                    )
                )

        # TODO DW: also save local solutions?
        #  (need to track fvals or re-evaluate)

        return result

    def _run_scatter_searches(
        self,
        refsets: List[RefSet],
    ) -> List[ESSOptimizer]:
        """Start all scatter searches in different processes."""
        # set default value of max_eval if not present.
        #  only set it on a copy, as the original dicts may be re-used
        #  for different optimization problems.
        # reasonable value proposed in [VillaverdeEge2012]:
        #  2.5 < tau < 3.5, default: 2.5
        ess_init_args = [
            dict(
                {
                    'max_eval': int(
                        10**2.5 * refsets[0].evaluator.problem.dim
                    )
                },
                **ess_kwargs,
            )
            for ess_kwargs in self.ess_init_args
        ]

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(len(self.ess_init_args)) as pool:
            ess_optimizers = pool.starmap(
                self._run_single_ess,
                (
                    [ess_kwargs, refset]
                    for (ess_kwargs, refset) in zip(ess_init_args, refsets)
                ),
                chunksize=1,
            )
        return list(ess_optimizers)

    def _run_single_ess(
        self,
        ess_kwargs,
        refset: RefSet,
    ) -> ESSOptimizer:
        """
        Run ESS.

        Helper for `starmap`.
        """
        # different random seeds per process?
        np.random.seed((os.getpid() * int(time.time() * 1000)) % 2**32)

        ess = ESSOptimizer(**ess_kwargs)
        ess.minimize(refset=refset)
        return ess

    def _keep_going(self, i_eval) -> bool:
        """Check exit criteria.

        Returns
        -------
        ``True`` if none of the exit criteria is met, ``False`` otherwise.
        """
        # TODO DW which further stopping criteria: gtol, fatol, frtol?

        # elapsed iterations
        if self.i_iter >= self.max_iter:
            self.exit_flag = ESSExitFlag.MAX_ITER
            return False

        # elapsed time
        if time.time() - self.starttime >= self.max_walltime_s:
            self.exit_flag = ESSExitFlag.MAX_TIME
            return False

        return True

    def _maybe_update_global_best(self, x, fx):
        """Update the global best value if the provided value is better."""
        if fx < self.fx_best:
            self.x_best = x[:]
            self.fx_best = fx

    def _update_refsets(
        self, refsets: List[RefSet], evaluator: FunctionEvaluator
    ):
        """
        Update refsets.

        Create new refsets based on the combined final refsets of the previous
        CESS iteration.

        Updates ``refsets`` in place.
        """
        # gather final refset entries
        x = np.vstack([refset.x for refset in refsets])
        fx = np.concatenate([refset.fx for refset in refsets])

        # reset function evaluation counter
        evaluator.n_eval = 0
        evaluator.n_eval_round = 0

        for i, ess_init_args in enumerate(self.ess_init_args):
            refsets[i] = RefSet(
                dim=ess_init_args['dim_refset'], evaluator=evaluator
            )
            refsets[i].initialize_from_array(x_diverse=x, fx_diverse=fx)
            refsets[i].sort()

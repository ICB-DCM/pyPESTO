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
    """
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

    .. [VillaverdeEge2012] 'A cooperative strategy for parameter estimation in
       large scale systems biology models', Villaverde, A.F., Egea,
       J.A. & Banga, J.R. BMC Syst Biol 2012, 6, 75.
       https://doi.org/10.1186/1752-0509-6-75

    Attributes
    ----------
    TODO
    """

    def __init__(
        self,
        max_n_eval: int,
        ess_init_args: List[Dict],
        max_iter: int,
    ):
        """Construct.

        Parameters
        ----------
        max_n_eval:
            TODO
        ess_init_args:
            List of argument dictionaries passed to
            :func:`ESSOptimizer.__init__`. The length of this list is the
            number of parallel ESS processes.
            Resource limits such as ``max_eval`` apply to a single CESS
            iteration, not to the full search.
        max_iter:
            Maximum number of CESS iterations.
        """
        self.max_n_eval = max_n_eval
        self.ess_init_args = ess_init_args
        self.max_iter = max_iter
        # Overall best parameters found so far
        self.x_best: Optional[np.array] = None
        # Overall best function value found so far
        self.fx_best: float = np.inf
        self.starttime = None
        self.i_iter = None

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
        self.starttime = time.time()

        # TODO as option --- used where??
        # [VillaverdeEge2012] Eq. 8
        # default value proposed in [VillaverdeEge2012]: 2.5 < tau < 3.5
        self.tau = np.log10(self.max_n_eval / problem.dim)
        self.i_iter = 0

        refsets = [None] * len(self.ess_init_args)

        evaluator = FunctionEvaluator(
            problem=problem,
            startpoint_method=startpoint_method,
            n_threads=1,
        )

        while True:
            logger.info("-" * 50)
            self._report_iteration()
            self.i_iter += 1

            # start ESS in all processes
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(len(self.ess_init_args)) as pool:
                # TODO rename - ess
                results = pool.starmap(
                    self._run_ess,
                    (
                        [ess_kwargs, problem, startpoint_method, refset]
                        for (ess_kwargs, refset) in zip(
                            self.ess_init_args, refsets
                        )
                    ),
                    chunksize=1,
                )
            # collect refsets from the different ESS runs
            refsets = [result.refset for result in results]

            # update best values from ESS results
            for result in results:
                self._maybe_update_global_best(result.x_best, result.fx_best)

            if not self._keep_going(i_eval=evaluator.n_eval):
                break

            # gather final refsets
            x = np.vstack([refset.x for refset in refsets])
            fx = np.concatenate([refset.fx for refset in refsets])

            # create new refsets based on the combined previous final refsets
            for i, ess_init_args in enumerate(self.ess_init_args):
                refsets[i] = RefSet(
                    dim=ess_init_args['dim_refset'], evaluator=evaluator
                )
                refsets[i].initialize_from_array(x_diverse=x, fx_diverse=fx)
                refsets[i].sort()

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

    def _run_ess(
        self,
        ess_kwargs,
        problem: Problem,
        startpoint_method: StartpointMethod,
        refset: RefSet,
    ):
        """
        Run ESS.

        Helper for `starmap`.
        """
        # different random seeds per prcess?
        np.random.seed((os.getpid() * int(time.time() * 1000)) % 2**32)

        ess = ESSOptimizer(**ess_kwargs)
        ess.minimize(problem, startpoint_method, refset=refset)
        return ess

    def _keep_going(self, i_eval) -> bool:
        """Check exit criteria.

        Returns
        -------
        ``True`` if not of the exit criteria is met, ``False`` otherwise.
        """
        # TODO DW which further stopping criteria: gtol, fatol, frtol?

        if self.i_iter >= self.max_iter:
            self.exit_flag = ESSExitFlag.MAX_ITER
            return False

        if i_eval >= self.max_n_eval:
            self.exit_flag = ESSExitFlag.MAX_EVAL
            return False

        return True

    def _maybe_update_global_best(self, x, fx):
        """Update the global best value if the provided value is better."""
        if fx < self.fx_best:
            self.x_best = x[:]
            self.fx_best = fx

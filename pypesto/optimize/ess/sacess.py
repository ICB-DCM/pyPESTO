"""Self-adaptive cooperative enhanced scatter search (SACESS)."""
import logging
import os
import time
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Tuple

import numpy as np

import pypesto
from pypesto.startpoint import StartpointMethod

from ..optimize import Problem
from .ess import ESSExitFlag, ESSOptimizer
from .function_evaluator import FunctionEvaluator
from .refset import RefSet

__all__ = ["SacessOptimizer"]

logger = logging.getLogger(__name__)


class SacessOptimizer:
    """SACESS optimizer.

    A shared-memory-based implementation of the SaCeSS algorithm presented in
    [PenasGon2017]_. Multiple processes (`workers`) run consecutive ESSs in
    parallel. After each ESS run, depending on the outcome, there is a chance
    of exchanging good parameters, and changing ESS hyperparameters to those of
    the most promising worker.
    """

    def __init__(
        self,
        ess_init_args: List[Dict[str, Any]],
        max_walltime_s: float = np.inf,
    ):
        """Construct.

        Parameters
        ----------
        ess_init_args:
            List of argument dictionaries passed to
            :func:`ESSOptimizer.__init__`. Each entry corresponds to one worker
            process. I.e., the length of this list is the number of ESSs.
            Ideally, this list contains some more conservative and some more
            aggressive configurations.
            Resource limits such as ``max_eval`` apply to a single CESS
            iteration, not to the full search.
        max_walltime_s:
            Maximum walltime in seconds. Will only be checked between local
            optimizations and other simulations, and thus, may be exceeded by
            the duration of a local search. Defaults to no limit.
        """
        self.num_workers = len(ess_init_args)
        if self.num_workers < 2:
            raise ValueError(
                f"{self.__class__.__name__} needs at least 2 workers."
            )
        self.ess_init_args = ess_init_args
        self.max_walltime_s = max_walltime_s
        self.exit_flag = ESSExitFlag.DID_NOT_RUN

    def minimize(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod,
    ):
        """Solve the given optimization problem."""
        logging.info(
            f"Running sacess with {self.num_workers} "
            f"workers: {self.ess_init_args}"
        )

        # shared memory manager to handle shared state
        # (simulates the sacess manager process)
        with Manager() as shmem_manager:
            sacess_manager = SacessManager(
                shmem_manager=shmem_manager,
                ess_options=self.ess_init_args,
                dim=problem.dim,
            )
            # create workers
            workers = [
                SacessWorker(
                    manager=sacess_manager,
                    ess_kwargs=ess_kwargs,
                    worker_idx=i,
                    max_walltime_s=self.max_walltime_s,
                )
                for i, ess_kwargs in enumerate(self.ess_init_args)
            ]
            # launch worker processes
            worker_processes = [
                Process(
                    name=f"sacess-worker-{i}",
                    target=_run_worker,
                    args=(worker, problem, startpoint_method),
                )
                for i, worker in enumerate(workers)
            ]
            for p in worker_processes:
                p.start()
            # wait for finish
            for p in worker_processes:
                p.join()

            # TODO: where are the good solutions recorded?
            logging.info(
                f"---- global best {sacess_manager._best_known_fx.value}"
            )

        return self._create_result(problem)

    def _create_result(self, problem: Problem) -> pypesto.Result:
        import os

        from pypesto.store.read_from_hdf5 import read_result

        result = None
        for worker_idx in range(self.num_workers):
            tmp_result = read_result(
                f"sacess-{worker_idx}_tmp.h5", problem=False, optimize=True
            )
            os.remove(f"sacess-{worker_idx}_tmp.h5")
            if result is None:
                result = tmp_result
            else:
                result.optimize_result.append(
                    tmp_result.optimize_result,
                    sort=False,
                    prefix=f"{worker_idx}-",
                )
        result.optimize_result.sort()

        return result


class SacessManager:
    """The Sacess manager.

    Manages shared memory of a SACESS run. Loosely corresponds to the manager
    process in [PenasGon2017]_.
    """

    def __init__(
        self,
        shmem_manager: SyncManager,
        ess_options: List[Dict[str, Any]],
        dim: int,
    ):
        self._num_workers = len(ess_options)
        self._ess_options = [shmem_manager.dict(o) for o in ess_options]
        self._best_known_fx = shmem_manager.Value("d", np.inf)
        self._best_known_x = shmem_manager.Array("d", [np.nan] * dim)
        self._best_settings = shmem_manager.dict()
        self._scores = shmem_manager.Array("d", range(self._num_workers))
        # Number of rejected solutions received from workers since last
        #  adaptation of epsilon
        self._rejections = shmem_manager.Value("i", 0)
        # initial threshold for ...
        self._rejection_threshold = shmem_manager.Value("d", 0.1)
        self._lock = shmem_manager.RLock()

        # scores of the workers, ordered by worker-index
        # initial score is the worker index
        self._worker_scores = shmem_manager.Array(
            "d", range(self._num_workers)
        )
        self._worker_comms = shmem_manager.Array(
            "d", [0.0] * self._num_workers
        )
        self.logger = logging.getLogger()

    def get_best_solution(self) -> Tuple[np.array, float]:
        with self._lock:
            return np.array(self._best_known_x), self._best_known_fx.value

    def get_best_settings(self):
        with self._lock:
            # send settings of the best performing ESS to the workers that requested it
            # send newsettings[np.argmax(self.scores)]
            # TODO, not neessarily from best val, but high potential
            return self._best_settings.value

    def submit_solution(
        self,
        x: np.array,
        fx: float,
        sender_idx: int,
        elapsed_time_s: float,
        ess_kwargs: Dict[str, Any],
    ):
        with self._lock:
            # cooperation step
            # solution improves best value by at least a factor of ...
            if (
                self._best_known_fx.value - fx
            ) / self._best_known_fx.value < self._rejection_threshold.value or (
                np.isfinite(fx) and not np.isfinite(self._best_known_fx.value)
            ):
                self.logger.debug(
                    f"Accepted solution from worker {sender_idx}: {fx}."
                )
                # accept
                self._best_known_fx.value = fx
                self._best_known_x.value = x
                self._best_settings.value = ess_kwargs
                self._worker_comms[sender_idx] += 1

                # TODO unclear why a longer duration since the last accepted
                #  solution gives a higher score
                self._worker_scores[sender_idx] = (
                    self._worker_comms[sender_idx] * elapsed_time_s
                )
            else:
                # reject solution
                self._rejections.value += 1
                self.logger.debug(
                    f"Rejected solution from worker {sender_idx} (total rejections: {self._rejections.value})."
                )
                # adapt acceptance threshold if too many solutions have been
                #  rejected
                if self._rejections.value > self._num_workers:
                    self._rejection_threshold.value /= 2
                    self.logger.debug(
                        "Lowered acceptance threshold to "
                        f"{self._rejection_threshold.value}."
                    )
                    self._rejections.value = 0


class SacessWorker:
    """A SACESS worker.

    Repeatedly runs ESSs and exchanges information with a SacessManager.
    Corresponds to a worker process in [PenasGon2017]_.
    """

    def __init__(
        self,
        manager: SacessManager,
        ess_kwargs: Dict[str, Any],
        worker_idx: int,
        max_walltime_s: float = np.inf,
    ):
        self.manager = manager
        self.worker_idx = worker_idx
        self.best_known_fx = np.inf
        # number of received solution since the last one was sent to the manager
        self.n_received_solutions = 0
        self.iter_solver = 0
        # number of function evaluations (?) since the last solution was sent to the manager
        self.neval = 0
        self.ess_kwargs = ess_kwargs
        self.acceptance_threshold = 1e-6  # TODO
        self.n_sent_solutions = 0
        self.max_walltime_s = max_walltime_s
        self.logger = logging.getLogger(str(os.getpid()))
        self.manager.logger = self.logger

    def run(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod,
    ):
        evaluator = FunctionEvaluator(
            problem=problem,
            startpoint_method=startpoint_method,
            n_threads=1,
        )
        self.start_time = time.time()
        # create refset from ndiverse
        refset = RefSet(dim=self.ess_kwargs['dim_refset'], evaluator=evaluator)
        refset.initialize_random(n_diverse=self.ess_kwargs['n_diverse'])
        i_iter = 0
        while self._keep_going():
            # run standard ESS
            ess = ESSOptimizer(**self.ess_kwargs)
            ess.logger.setLevel(logging.WARNING)
            ess_results = ess.minimize(
                problem=problem,
                startpoint_method=startpoint_method,
                refset=refset,
            )
            self.logger.debug(
                f"#{self.worker_idx}: ESS finished with best "
                f"f(x)={ess.fx_best}"
            )
            from ...store.save_to_hdf5 import write_result

            write_result(
                ess_results,
                f"sacess-{self.worker_idx}_tmp.h5",
                overwrite=True,
                optimize=True,
            )
            # check if the last local ESS best solution is sufficiently better
            # than the sacess-wide best solution
            self.maybe_update_best(ess.x_best, ess.fx_best, self.ess_kwargs)
            self.best_known_fx = min(ess.fx_best, self.best_known_fx)

            # cooperation step
            # try to obtain a new solution from manager
            recv_x, recv_fx = self.manager.get_best_solution()
            self.logger.debug(
                f"Worker {self.worker_idx} received solution {recv_fx} (known best: {self.best_known_fx})."
            )
            if recv_fx < self.best_known_fx:
                logging.warning(
                    f"Worker {self.worker_idx} received better solution."
                )
                self.best_known_fx = recv_x
                self.n_received_solutions += 1
                self.replace_solution(refset, x=recv_x, fx=recv_fx)

            # adaptive step
            # Update ESS settings if we received way more solutions than we
            # sent
            if (
                self.n_received_solutions > 10 * self.n_sent_solutions + 20
                and self.neval > problem.dim * 5000
            ):
                self.ess_kwargs = self.manager.get_best_settings()
                self.logger.debug(
                    f"Updated settings on worker {self.worker_idx} to "
                    f"{self.ess_kwargs}"
                )

            self.logger.info(
                f"sacess worker {self.worker_idx} iteration {i_iter} "
                f"(best: {self.best_known_fx})."
            )

            i_iter += 1

    def maybe_update_best(self, x, fx, ess_kwargs):
        self.logger.debug(
            f"Worker {self.worker_idx} maybe sending solution {fx}. "
            f"{self.best_known_fx} {fx} {(self.best_known_fx - fx) / fx} {self.acceptance_threshold}"
        )

        # solution improves best value by at least a factor of ...
        if (self.best_known_fx - fx) / fx < self.acceptance_threshold or (
            np.isfinite(fx) and not np.isfinite(self.best_known_fx)
        ):
            self.logger.debug(
                f"Worker {self.worker_idx} sending solution {fx}."
            )
            self.n_sent_solutions += 1
            self.best_known_fx = fx

            # send best_known_fx to manager
            self.neval = 0
            self.n_received_solutions = 0
            elapsed_time = time.time() - self.start_time
            self.manager.submit_solution(
                x=x,
                fx=fx,
                ess_kwargs=ess_kwargs,
                sender_idx=self.worker_idx,
                elapsed_time_s=elapsed_time,
            )

    def replace_solution(self, refset: RefSet, x: np.array, fx: float):
        """Replace the global refset member by the given solution."""
        # (page 8, top)
        # on first call, replace the worst solution
        # TODO: first call of each ess or first call on worker?
        if "cooperative_solution" not in refset.attributes:
            label = np.zeros(shape=refset.dim)
            label[np.argmax(refset.fx)] = 1
            refset.add_attribute("cooperative_solution", label)

        # mark that one "cooperative solution"
        # subsequently replace the cooperative solution
        else:
            refset.update(
                i=np.argwhere(refset.attributes["cooperative_solution"]),
                x=x,
                fx=fx,
            )

    def _keep_going(self):
        """Check exit criteria.

        Returns
        -------
        ``True`` if none of the exit criteria is met, ``False`` otherwise.
        """
        # elapsed time
        if time.time() - self.start_time >= self.max_walltime_s:
            self.exit_flag = ESSExitFlag.MAX_TIME
            self.logger.debug(
                f"Max walltime ({self.max_walltime_s}s) exceeded."
            )
            return False

        return True


def _run_worker(
    worker: SacessWorker,
    problem: Problem,
    startpoint_method: StartpointMethod,
):
    """Run the given SACESS worker.

    Helper function as entrypoint for sacess worker processes.
    """
    # create a thread-local logger
    global logger
    logger = logging.getLogger(__name__)

    return worker.run(problem=problem, startpoint_method=startpoint_method)

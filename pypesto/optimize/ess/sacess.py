"""Self-adaptive cooperative enhanced scatter search (SACESS)."""
import logging
import os
import time
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Tuple

import numpy as np

import pypesto

from ...startpoint import StartpointMethod
from ...store.read_from_hdf5 import read_result
from ...store.save_to_hdf5 import write_result
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
        logging.debug(
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

            logging.info(
                f"Global best: {sacess_manager.get_best_solution()[1]}"
            )

        return self._create_result(problem)

    def _create_result(self, problem: Problem) -> pypesto.Result:
        """Create result object.

        Creates an overall Result object from the results saved by the workers.
        """
        # gather results from workers and delete temporary result files
        result = None
        for worker_idx in range(self.num_workers):
            tmp_result_filename = f"sacess-{worker_idx}_tmp.h5"
            tmp_result = read_result(
                tmp_result_filename, problem=False, optimize=True
            )
            os.remove(tmp_result_filename)
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

    Attributes
    ----------
    _num_workers: Number of workers
    _ess_options: ESS options for each worker
    _best_known_fx: Best objective value encountered so far
    _best_known_x: Parameters corresponding to ``_best_known_fx``
    _worker_scores: Performance score of the different workers (the higher, the
        more promising the respective worker is considered)
    _worker_comms: Number of communications received from the individual
        workers
    _rejections: Number of rejected solutions received from workers since last
        adaptation of ``_rejection_threshold``.
    _rejection_threshold: Threshold for relative objective improvements that
        incoming solutions have to pass to be accepted
    _lock: Lock for accessing shared state.
    _logger: A logger instance
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
        self._rejections = shmem_manager.Value("i", 0)
        # initial value from [PenasGon2017]_ p.9
        self._rejection_threshold = shmem_manager.Value("d", 0.1)
        # scores of the workers, ordered by worker-index
        # initial score is the worker index
        self._worker_scores = shmem_manager.Array(
            "d", range(self._num_workers)
        )
        self._worker_comms = shmem_manager.Array("i", [0] * self._num_workers)
        self._lock = shmem_manager.RLock()
        self._logger = logging.getLogger()

    def get_best_solution(self) -> Tuple[np.array, float]:
        """Get the best objective value and corresponding parameters."""
        with self._lock:
            return np.array(self._best_known_x), self._best_known_fx.value

    def reconfigure_worker(self, worker_idx: int) -> Dict:
        """Reconfigure the given worker.

        Updates the ESS options for the given worker to those of the worker at
        the top of the scoreboard and returns those settings.
        """
        with self._lock:
            leader_options = self._ess_options[np.argmax(self._worker_scores)]
            self._ess_options[worker_idx] = leader_options
            return leader_options

    def submit_solution(
        self,
        x: np.array,
        fx: float,
        sender_idx: int,
        elapsed_time_s: float,
    ):
        """Submit a solution.

        To be called by a worker.

        Parameters
        ----------
        x: Model parameters
        fx: Objective value corresponding to ``x``
        sender_idx: Index of the worker submitting the results.
        elapsed_time_s: Elapsed time since the beginning of the sacess run.
        """
        with self._lock:
            # cooperation step
            # solution improves best value by at least a factor of ...
            if (
                # initially _best_known_fx is NaN
                (
                    np.isfinite(fx)
                    and not np.isfinite(self._best_known_fx.value)
                )
                # avoid division by 0. just accept any improvement if best
                # known value is 0.
                or (self._best_known_fx.value == 0 and fx < 0)
                or (
                    fx < self._best_known_fx.value
                    and abs(
                        (self._best_known_fx.value - fx)
                        / self._best_known_fx.value
                    )
                    < self._rejection_threshold.value
                )
            ):
                self._logger.debug(
                    f"Accepted solution from worker {sender_idx}: {fx}."
                )
                # accept
                self._best_known_fx.value = fx
                self._best_known_x.value = x
                self._worker_comms[sender_idx] += 1

                self._worker_scores[sender_idx] = (
                    self._worker_comms[sender_idx] * elapsed_time_s
                )
            else:
                # reject solution
                self._rejections.value += 1
                self._logger.debug(
                    f"Rejected solution from worker {sender_idx} "
                    f"(total rejections: {self._rejections.value})."
                )
                # adapt acceptance threshold if too many solutions have been
                #  rejected
                if self._rejections.value > self._num_workers:
                    self._rejection_threshold.value /= 2
                    self._logger.debug(
                        "Lowered acceptance threshold to "
                        f"{self._rejection_threshold.value}."
                    )
                    self._rejections.value = 0


class SacessWorker:
    """A SACESS worker.

    Repeatedly runs ESSs and exchanges information with a SacessManager.
    Corresponds to a worker process in [PenasGon2017]_.

    Attributes
    ----------
    _manager: The sacess manager this worker is working for.
    _worker_idx: Index of this worker.
    _best_known_fx: Best objective value known to this worker (obtained on its
        own or received from the manager).
    _n_received_solutions: Number of solutions received by this worker since
        the last one was sent to the manager.
    _neval: Number of objective evaluations since the last solution was sent
        to the manager.
    _ess_kwargs: ESSOptimizer options for this worker (may get updated during
        the self-adaptive step).
    _acceptance_threshold: Minimum relative improvement of the objective
        compared to the best known value to be eligible for submission to the
        Manager.
    _n_sent_solutions: Number of solutions sent to the Manager.
    _max_walltime_s: Walltime limit.
    _logger: A Logger instance.
    """

    def __init__(
        self,
        manager: SacessManager,
        ess_kwargs: Dict[str, Any],
        worker_idx: int,
        max_walltime_s: float = np.inf,
    ):
        self._manager = manager
        self._worker_idx = worker_idx
        self._best_known_fx = np.inf
        self._n_received_solutions = 0
        self._neval = 0
        self._ess_kwargs = ess_kwargs
        self._acceptance_threshold = 1e-6  # TODO
        self._n_sent_solutions = 0
        self._max_walltime_s = max_walltime_s
        self._start_time = None
        self._logger = logging.getLogger(str(os.getpid()))
        # Set the manager logger to one created within the current process
        self._manager._logger = self._logger

    def run(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod,
    ):
        """Start the worker."""
        self._logger.debug(f"#{self._worker_idx} starting.")

        evaluator = FunctionEvaluator(
            problem=problem,
            startpoint_method=startpoint_method,
            n_threads=1,
        )
        self._start_time = time.time()
        # create refset from ndiverse
        refset = RefSet(
            dim=self._ess_kwargs['dim_refset'], evaluator=evaluator
        )
        refset.initialize_random(n_diverse=self._ess_kwargs['n_diverse'])
        i_iter = 0
        while self._keep_going():
            # run standard ESS
            ess_kwargs = self._ess_kwargs.copy()
            # account for sacess walltime limit
            ess_kwargs['max_walltime_s'] = min(
                ess_kwargs.get('max_walltime_s', np.inf),
                self._max_walltime_s - (time.time() - self._start_time),
            )
            ess = ESSOptimizer(**ess_kwargs)
            ess.logger.setLevel(logging.WARNING)
            ess_results = ess.minimize(
                problem=problem,
                startpoint_method=startpoint_method,
                refset=refset,
            )
            self._logger.debug(
                f"#{self._worker_idx}: ESS finished with best "
                f"f(x)={ess.fx_best}"
            )

            # TODO accumulate results from all ESSs
            write_result(
                ess_results,
                f"sacess-{self._worker_idx}_tmp.h5",
                overwrite=True,
                optimize=True,
            )
            # check if the last local ESS best solution is sufficiently better
            # than the sacess-wide best solution
            self.maybe_update_best(ess.x_best, ess.fx_best)
            self._best_known_fx = min(ess.fx_best, self._best_known_fx)

            # cooperation step
            # try to obtain a new solution from manager
            recv_x, recv_fx = self._manager.get_best_solution()
            self._logger.debug(
                f"Worker {self._worker_idx} received solution {recv_fx} "
                f"(known best: {self._best_known_fx})."
            )
            if recv_fx < self._best_known_fx:
                logging.warning(
                    f"Worker {self._worker_idx} received better solution."
                )
                self._best_known_fx = recv_x
                self._n_received_solutions += 1
                self.replace_solution(refset, x=recv_x, fx=recv_fx)

            # adaptive step
            # Update ESS settings if we received way more solutions than we
            # sent
            if (
                self._n_received_solutions > 10 * self._n_sent_solutions + 20
                and self._neval > problem.dim * 5000
            ):
                self._ess_kwargs = self._manager.reconfigure_worker(
                    self._worker_idx
                )
                self._logger.debug(
                    f"Updated settings on worker {self._worker_idx} to "
                    f"{self._ess_kwargs}"
                )

            self._logger.info(
                f"sacess worker {self._worker_idx} iteration {i_iter} "
                f"(best: {self._best_known_fx})."
            )

            i_iter += 1

    def maybe_update_best(self, x: np.array, fx: float):
        """Maybe update the best known solution and send it to the manager."""
        self._logger.debug(
            f"Worker {self._worker_idx} maybe sending solution {fx}. "
            f"best known: {self._best_known_fx}, "
            f"rel change: {(self._best_known_fx - fx) / fx}, "
            f"threshold: {self._acceptance_threshold}"
        )

        # solution improves best value by at least a factor of ...
        if (
            np.isfinite(fx)
            and not np.isfinite(self._best_known_fx)
            or (self._best_known_fx == 0 and fx < 0)
            or (
                fx < self._best_known_fx
                and abs((self._best_known_fx - fx) / fx)
                < self._acceptance_threshold
            )
        ):
            self._logger.debug(
                f"Worker {self._worker_idx} sending solution {fx}."
            )
            self._n_sent_solutions += 1
            self._best_known_fx = fx

            # send best_known_fx to manager
            self._neval = 0
            self._n_received_solutions = 0
            elapsed_time = time.time() - self._start_time
            self._manager.submit_solution(
                x=x,
                fx=fx,
                sender_idx=self._worker_idx,
                elapsed_time_s=elapsed_time,
            )

    @staticmethod
    def replace_solution(refset: RefSet, x: np.array, fx: float):
        """Replace the global refset member by the given solution."""
        # [PenasGon2017]_ page 8, top
        if "cooperative_solution" not in refset.attributes:
            label = np.zeros(shape=refset.dim)
            # on first call, mark the worst solution as "cooperative solution"
            label[np.argmax(refset.fx)] = 1
            refset.add_attribute("cooperative_solution", label)

        # replace the cooperative solution
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
        if time.time() - self._start_time >= self._max_walltime_s:
            self.exit_flag = ESSExitFlag.MAX_TIME
            self._logger.debug(
                f"Max walltime ({self._max_walltime_s}s) exceeded."
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

    # different random seeds per process
    np.random.seed((os.getpid() * int(time.time() * 1000)) % 2**32)

    return worker.run(problem=problem, startpoint_method=startpoint_method)

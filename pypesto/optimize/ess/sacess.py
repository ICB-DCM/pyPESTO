"""Self-adaptive cooperative enhanced scatter search (SACESS)."""
import itertools
import logging
import logging.handlers
import multiprocessing
import os
import time
from math import ceil, sqrt
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid1
from warnings import warn

import numpy as np

import pypesto

from ...startpoint import StartpointMethod
from ...store.read_from_hdf5 import read_result
from ...store.save_to_hdf5 import write_result
from ..optimize import Problem
from .ess import ESSExitFlag, ESSOptimizer
from .function_evaluator import create_function_evaluator
from .refset import RefSet

__all__ = [
    "SacessOptimizer",
    "get_default_ess_options",
    "SacessFidesFactory",
]

logger = logging.getLogger(__name__)


class SacessOptimizer:
    """SACESS optimizer.

    A shared-memory-based implementation of the SaCeSS algorithm presented in
    :footcite:t:`PenasGon2017`. Multiple processes (`workers`) run
    :class:`enhanced scatter searches (ESSs) <ESSOptimizer>` in parallel.
    After each ESS iteration, depending on the outcome, there is a chance
    of exchanging good parameters, and changing ESS hyperparameters to those of
    the most promising worker. See :footcite:t:`PenasGon2017` for details.

    :class:`SacessOptimizer` can be used with or without a local optimizer, but
    it is highly recommended to use one.

    .. footbibliography::

    Attributes
    ----------
    histories:
        List of the histories of the best values/parameters
        found by each worker. (Monotonously decreasing objective values.)
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        ess_init_args: Optional[List[Dict[str, Any]]] = None,
        max_walltime_s: float = np.inf,
        sacess_loglevel: int = logging.INFO,
        ess_loglevel: int = logging.WARNING,
        tmpdir: Union[Path, str] = None,
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
            Mutually exclusive with ``num_workers``.
            Recommended default settings can be obtained from
            :func:`get_default_ess_options`.
        num_workers:
            Number of workers to be used. If this argument is given,
            (different) default ESS settings will be used for each worker.
            Mutually exclusive with ``ess_init_args``.
            See :func:`get_default_ess_options` for details on the default
            settings.
        max_walltime_s:
            Maximum walltime in seconds. It will only be checked between local
            optimizations and other simulations, and thus, may be exceeded by
            the duration of a local search. Defaults to no limit.
            Note that in order to impose the wall time limit also on the local
            optimizer, the user has to provide a wrapper function similar to
            :meth:`SacessFidesFactory.__call__`.
        ess_loglevel:
            Loglevel for ESS runs.
        sacess_loglevel:
            Loglevel for SACESS runs.
        tmpdir:
            Directory for temporary files. This defaults to a directory in the
            current working directory named
            ``SacessOptimizerTemp-{random suffix}``.
            When setting this option, make sure any optimizers running in
            parallel have unique `tmpdir`s.
        """
        if (num_workers is None and ess_init_args is None) or (
            num_workers is not None and ess_init_args is not None
        ):
            raise ValueError(
                "Exactly one of `num_workers` or `ess_init_args` "
                "has to be provided."
            )

        self.num_workers = num_workers or len(ess_init_args)
        if self.num_workers < 2:
            raise ValueError(
                f"{self.__class__.__name__} needs at least 2 workers."
            )
        self.ess_init_args = ess_init_args
        self.max_walltime_s = max_walltime_s
        self.exit_flag = ESSExitFlag.DID_NOT_RUN
        self.ess_loglevel = ess_loglevel
        self.sacess_loglevel = sacess_loglevel
        logger.setLevel(self.sacess_loglevel)

        self._tmpdir = tmpdir
        if self._tmpdir is None:
            while self._tmpdir is None or self._tmpdir.exists():
                self._tmpdir = Path(f"SacessOptimizerTemp-{str(uuid1())[:8]}")
        self._tmpdir = Path(self._tmpdir).absolute()
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self.histories: Optional[
            list["pypesto.history.memory.MemoryHistory"]
        ] = None

    def minimize(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod = None,
    ) -> pypesto.Result:
        """Solve the given optimization problem.

        Note that if this function is called from a multithreaded program (
        multiple threads running at the time of calling this function) and
        the :mod:`multiprocessing` `start method` is set to ``fork``, there is
        a good chance for deadlocks. Postpone spawning threads until after
        `minimize` or change the *start method* to ``spawn``.

        Parameters
        ----------
        problem:
            Minimization problem.
            :meth:`Problem.startpoint_method` will be used to sample random
            points. `SacessOptimizer` will deal with non-evaluable points.
            Therefore, using :class:`pypesto.startpoint.CheckedStartpoints`
            with ``check_fval=True`` or ``check_grad=True`` is not recommended
            since it would create significant overhead.

        startpoint_method:
            Method for choosing starting points.
            **Deprecated. Use ``problem.startpoint_method`` instead.**

        Returns
        -------
        Result object with optimized parameters in
        :attr:`pypesto.Result.optimize_result`.
        Results are sorted by objective. At least the best parameters are
        included. Additional results may be included - this is subject to
        change.
        """
        if startpoint_method is not None:
            warn(
                "Passing `startpoint_method` directly is deprecated, use `problem.startpoint_method` instead.",
                DeprecationWarning,
            )

        start_time = time.time()
        logger.debug(
            f"Running {self.__class__.__name__} with {self.num_workers} "
            f"workers: {self.ess_init_args}"
        )
        ess_init_args = self.ess_init_args or get_default_ess_options(
            num_workers=self.num_workers, dim=problem.dim
        )

        logging_handler = logging.StreamHandler()
        logging_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s %(name)s %(levelname)-8s %(message)s'
            )
        )
        logging_thread = logging.handlers.QueueListener(
            multiprocessing.Queue(-1), logging_handler
        )

        # shared memory manager to handle shared state
        # (simulates the sacess manager process)
        with Manager() as shmem_manager:
            sacess_manager = SacessManager(
                shmem_manager=shmem_manager,
                ess_options=ess_init_args,
                dim=problem.dim,
            )
            # create workers
            workers = [
                SacessWorker(
                    manager=sacess_manager,
                    ess_kwargs=ess_kwargs,
                    worker_idx=worker_idx,
                    max_walltime_s=self.max_walltime_s,
                    loglevel=self.sacess_loglevel,
                    ess_loglevel=self.ess_loglevel,
                    tmp_result_file=SacessWorker.get_temp_result_filename(
                        worker_idx, self._tmpdir
                    ),
                )
                for worker_idx, ess_kwargs in enumerate(ess_init_args)
            ]
            # launch worker processes
            worker_processes = [
                Process(
                    name=f"{self.__class__.__name__}-worker-{i:02d}",
                    target=_run_worker,
                    args=(
                        worker,
                        problem,
                        startpoint_method,
                        logging_thread.queue,
                    ),
                )
                for i, worker in enumerate(workers)
            ]
            for p in worker_processes:
                p.start()

            # start logging thread only AFTER starting the worker processes
            #  to prevent deadlocks
            logging_thread.start()

            # wait for finish
            # collect results
            histories = [
                sacess_manager._result_queue.get()
                for _ in range(self.num_workers)
            ]
            self.histories = histories
            for p in worker_processes:
                p.join()

        logging_thread.stop()
        result = self._create_result(problem)

        walltime = time.time() - start_time
        logger.info(
            f"{self.__class__.__name__} stopped after {walltime:3g}s with global best "
            f"{result.optimize_result[0].fval}."
        )

        return result

    def _create_result(self, problem: Problem) -> pypesto.Result:
        """Create result object.

        Creates an overall Result object from the results saved by the workers.
        """
        # gather results from workers and delete temporary result files
        result = None
        for worker_idx in range(self.num_workers):
            tmp_result_filename = SacessWorker.get_temp_result_filename(
                worker_idx, self._tmpdir
            )
            try:
                tmp_result = read_result(
                    tmp_result_filename, problem=False, optimize=True
                )
            except FileNotFoundError:
                # wait and retry, maybe the file wasn't found due to some filesystem latency issues
                time.sleep(5)
                tmp_result = read_result(
                    tmp_result_filename, problem=False, optimize=True
                )

            if result is None:
                result = tmp_result
            else:
                result.optimize_result.append(
                    tmp_result.optimize_result,
                    sort=False,
                    prefix=f"{worker_idx}-",
                )

        # delete temporary files only after successful consolidation
        for worker_idx in range(self.num_workers):
            filename = SacessWorker.get_temp_result_filename(
                worker_idx, self._tmpdir
            )
            os.remove(filename)
        # delete tmpdir if empty
        try:
            self._tmpdir.rmdir()
        except OSError:
            pass

        result.optimize_result.sort()

        result.problem = problem

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
        self._result_queue = shmem_manager.Queue()

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
            leader_options = self._ess_options[
                np.argmax(self._worker_scores)
            ].copy()
            for setting in ["local_n2", "balance", "dim_refset"]:
                if setting in leader_options:
                    self._ess_options[worker_idx][setting] = leader_options[
                        setting
                    ]
            return self._ess_options[worker_idx].copy()

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
        abs_change = fx - self._best_known_fx.value
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
                    and abs(abs_change / self._best_known_fx.value)
                    > self._rejection_threshold.value
                )
            ):
                # accept solution
                self._logger.debug(
                    f"Accepted solution from worker {sender_idx}: {fx}."
                )
                # accept
                if len(x) != len(self._best_known_x):
                    raise AssertionError(
                        f"Received solution with {len(x)} parameters, "
                        f"but expected {len(self._best_known_x)}."
                    )
                for i, xi in enumerate(x):
                    self._best_known_x[i] = xi
                self._best_known_fx.value = fx
                self._worker_comms[sender_idx] += 1
                self._worker_scores[sender_idx] = (
                    self._worker_comms[sender_idx] * elapsed_time_s
                )
            else:
                # reject solution
                self._rejections.value += 1

                self._logger.debug(
                    f"Rejected solution from worker {sender_idx} "
                    f"abs change: {abs_change} "
                    f"rel change: {abs(abs_change / self._best_known_fx.value):.4g} "
                    f"(threshold: {self._rejection_threshold.value}) "
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

    Runs ESSs and exchanges information with a SacessManager.
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
    _loglevel: Logging level for sacess
    _ess_loglevel: Logging level for ESS runs
    _tmp_result_file: Path of a temporary file to be created.
    """

    def __init__(
        self,
        manager: SacessManager,
        ess_kwargs: Dict[str, Any],
        worker_idx: int,
        max_walltime_s: float = np.inf,
        loglevel: int = logging.INFO,
        ess_loglevel: int = logging.WARNING,
        tmp_result_file: str = None,
    ):
        self._manager = manager
        self._worker_idx = worker_idx
        self._best_known_fx = np.inf
        self._n_received_solutions = 0
        self._neval = 0
        self._ess_kwargs = ess_kwargs
        self._acceptance_threshold = 0.005
        self._n_sent_solutions = 0
        self._max_walltime_s = max_walltime_s
        self._start_time = None
        self._loglevel = loglevel
        self._ess_loglevel = ess_loglevel
        self._logger = None
        self._tmp_result_file = tmp_result_file
        self._refset = None

    def run(
        self,
        problem: Problem,
        startpoint_method: StartpointMethod,
    ):
        self._start_time = time.time()

        self._logger.setLevel(self._loglevel)
        # Set the manager logger to one created within the current process
        self._manager._logger = self._logger

        self._logger.debug(
            f"#{self._worker_idx} starting " f"({self._ess_kwargs})."
        )

        evaluator = create_function_evaluator(
            problem,
            startpoint_method,
            n_procs=self._ess_kwargs.get('n_procs'),
            n_threads=self._ess_kwargs.get('n_threads'),
        )

        # create initial refset
        self._refset = RefSet(
            dim=self._ess_kwargs['dim_refset'], evaluator=evaluator
        )
        self._refset.initialize_random(
            n_diverse=max(
                self._ess_kwargs.get('n_diverse', 10 * problem.dim),
                self._refset.dim,
            )
        )

        ess_results = pypesto.Result(problem=problem)
        ess = self._setup_ess(startpoint_method)

        # run ESS until exit criteria are met, but start at least one iteration
        while self._keep_going() or ess.n_iter == 0:
            # perform one ESS iteration
            ess._do_iteration()

            # TODO maybe not in every iteration?
            # drop all but the 50 best results
            cur_ess_results = ess._create_result()
            ess_results.optimize_result.append(
                cur_ess_results.optimize_result,
                prefix=f"{self._worker_idx}_{ess.n_iter}_",
            )
            ess_results.optimize_result.list = (
                ess_results.optimize_result.list[:50]
            )
            if self._tmp_result_file:
                write_result(
                    ess_results,
                    self._tmp_result_file,
                    overwrite=True,
                    optimize=True,
                )
            # check if the best solution of the last local ESS is sufficiently
            # better than the sacess-wide best solution
            self.maybe_update_best(ess.x_best, ess.fx_best)
            self._best_known_fx = min(ess.fx_best, self._best_known_fx)

            self._cooperate()
            self._maybe_adapt(problem)

            self._logger.info(
                f"sacess worker {self._worker_idx} iteration {ess.n_iter} "
                f"(best: {self._best_known_fx})."
            )
        ess.history.finalize(exitflag=ess.exit_flag.name)
        self._manager._result_queue.put(ess.history)
        ess._report_final()

    def _setup_ess(self, startpoint_method: StartpointMethod) -> ESSOptimizer:
        """Run ESS."""
        ess_kwargs = self._ess_kwargs.copy()
        # account for sacess walltime limit
        ess_kwargs['max_walltime_s'] = min(
            ess_kwargs.get('max_walltime_s', np.inf),
            self._max_walltime_s - (time.time() - self._start_time),
        )

        ess = ESSOptimizer(**ess_kwargs)
        ess.logger = self._logger.getChild(
            f"sacess-{self._worker_idx:02d}-ess"
        )
        ess.logger.setLevel(self._ess_loglevel)

        ess._initialize_minimize(
            startpoint_method=startpoint_method, refset=self._refset
        )

        return ess

    def _cooperate(self):
        """Cooperation step."""
        # try to obtain a new solution from manager
        recv_x, recv_fx = self._manager.get_best_solution()
        self._logger.log(
            logging.DEBUG - 1,
            f"Worker {self._worker_idx} received solution {recv_fx} "
            f"(known best: {self._best_known_fx}).",
        )
        if recv_fx < self._best_known_fx or (
            not np.isfinite(self._best_known_fx) and np.isfinite(recv_fx)
        ):
            if not np.isfinite(recv_x).all():
                raise AssertionError(
                    f"Received non-finite parameters {recv_x}."
                )
            self._logger.debug(
                f"Worker {self._worker_idx} received better solution {recv_fx}."
            )
            self._best_known_fx = recv_fx
            self._n_received_solutions += 1
            self.replace_solution(self._refset, x=recv_x, fx=recv_fx)

    def _maybe_adapt(self, problem: Problem):
        """Perform adaptation step.

        Update ESS settings if conditions are met.
        """
        # Update ESS settings if we received way more solutions than we sent
        # Magic numbers from [PenasGon2017]_ algorithm 5
        if (
            self._n_received_solutions > 10 * self._n_sent_solutions + 20
            and self._neval > problem.dim * 5000
        ):
            self._ess_kwargs = self._manager.reconfigure_worker(
                self._worker_idx
            )
            self._refset.resize(self._ess_kwargs['dim_refset'])
            self._logger.debug(
                f"Updated settings on worker {self._worker_idx} to "
                f"{self._ess_kwargs}"
            )

    def maybe_update_best(self, x: np.array, fx: float):
        """Maybe update the best known solution and send it to the manager."""
        self._logger.debug(
            f"Worker {self._worker_idx} maybe sending solution {fx}. "
            f"best known: {self._best_known_fx}, "
            f"rel change: {(fx - self._best_known_fx) / fx:.4g}, "
            f"threshold: {self._acceptance_threshold}"
        )

        # solution improves best value by at least a factor of ...
        if (
            (np.isfinite(fx) and not np.isfinite(self._best_known_fx))
            or (self._best_known_fx == 0 and fx < 0)
            or (
                fx < self._best_known_fx
                and abs((self._best_known_fx - fx) / fx)
                > self._acceptance_threshold
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
            cooperative_solution_idx = np.argmax(refset.fx)
            label[cooperative_solution_idx] = 1
            refset.add_attribute("cooperative_solution", label)
        elif (
            cooperative_solution_idx := np.argwhere(
                refset.attributes["cooperative_solution"]
            )
        ).size == 0:
            # the attribute exists, but no member is marked as cooperative
            # solution. this may happen if we shrink the refset.
            cooperative_solution_idx = np.argmax(refset.fx)

        # replace the cooperative solution
        refset.update(
            i=cooperative_solution_idx,
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

    @staticmethod
    def get_temp_result_filename(
        worker_idx: int, tmpdir: Union[str, Path]
    ) -> str:
        return str(Path(tmpdir, f"sacess-{worker_idx:02d}_tmp.h5").absolute())


def _run_worker(
    worker: SacessWorker,
    problem: Problem,
    startpoint_method: StartpointMethod,
    log_process_queue: multiprocessing.Queue,
):
    """Run the given SACESS worker.

    Helper function as entrypoint for sacess worker processes.
    """
    # different random seeds per process
    np.random.seed((os.getpid() * int(time.time() * 1000)) % 2**32)

    # Forward log messages to logging process
    h = logging.handlers.QueueHandler(log_process_queue)
    worker._logger = logging.getLogger(multiprocessing.current_process().name)
    worker._logger.addHandler(h)

    return worker.run(problem=problem, startpoint_method=startpoint_method)


def get_default_ess_options(
    num_workers: int,
    dim: int,
    local_optimizer: Union[
        bool,
        "pypesto.optimize.Optimizer",
        Callable[..., "pypesto.optimize.Optimizer"],
    ] = True,
) -> List[Dict]:
    """Get default ESS settings for (SA)CESS.

    Returns settings for ``num_workers`` parallel scatter searches, combining
    more aggressive and more conservative configurations. Mainly intended for
    use with :class:`SacessOptimizer`. For details on the different options,
    see keyword arguments of :meth:`ESSOptimizer.__init__`.

    Setting appropriate values for ``n_threads`` and ``local_optimizer`` is
    left to the user. Defaults to single-threaded and no local optimizer.

    Based on https://bitbucket.org/DavidPenas/sacess-library/src/508e7ac15579104731cf1f8c3969960c6e72b872/src/method_module_fortran/eSS/parallelscattersearchfunctions.f90#lines-929

    Parameters
    ----------
    num_workers: Number of configurations to return.
    dim: Problem dimension (number of optimized parameters).
    local_optimizer: The local optimizer to use
        (see same argument in :class:`ESSOptimizer`), a boolean indicating
        whether to set the default local optimizer
        (currently :class:`FidesOptimizer`), a :obj:`Callable` returning an
        optimizer instance.
        The latter can be used to propagate walltime limits to the local
        optimizers. See :meth:`SacessFidesFactory.__call__` for an example.
    """
    min_dimrefset = 5

    def dim_refset(x):
        return max(min_dimrefset, ceil((1 + sqrt(4 * dim * x)) / 2))

    settings = [
        # settings for first worker
        {
            'dim_refset': dim_refset(10),
            'balance': 0.5,
            'local_n2': 10,
        },
        # for the remaining workers, cycle through these settings
        # 1
        {
            'dim_refset': dim_refset(1),
            'balance': 0.0,
            'local_n1': 1,
            'local_n2': 1,
        },
        # 2
        {
            'dim_refset': dim_refset(3),
            'balance': 0.0,
            'local_n1': 1000,
            'local_n2': 1000,
        },
        # 3
        {
            'dim_refset': dim_refset(5),
            'balance': 0.25,
            'local_n1': 10,
            'local_n2': 10,
        },
        # 4
        {
            'dim_refset': dim_refset(10),
            'balance': 0.5,
            'local_n1': 20,
            'local_n2': 20,
        },
        # 5
        {
            'dim_refset': dim_refset(15),
            'balance': 0.25,
            'local_n1': 100,
            'local_n2': 100,
        },
        # 6
        {
            'dim_refset': dim_refset(12),
            'balance': 0.25,
            'local_n1': 1000,
            'local_n2': 1000,
        },
        # 7
        {
            'dim_refset': dim_refset(7.5),
            'balance': 0.25,
            'local_n1': 15,
            'local_n2': 15,
        },
        # 8
        {
            'dim_refset': dim_refset(5),
            'balance': 0.25,
            'local_n1': 7,
            'local_n2': 7,
        },
        # 9
        {
            'dim_refset': dim_refset(2),
            'balance': 0.0,
            'local_n1': 1000,
            'local_n2': 1000,
        },
        # 10
        {
            'dim_refset': dim_refset(0.5),
            'balance': 0.0,
            'local_n1': 1,
            'local_n2': 1,
        },
        # 11
        {
            'dim_refset': dim_refset(1.5),
            'balance': 1.0,
            'local_n1': 1,
            'local_n2': 1,
        },
        # 12
        {
            'dim_refset': dim_refset(3.5),
            'balance': 1.0,
            'local_n1': 4,
            'local_n2': 4,
        },
        # 13
        {
            'dim_refset': dim_refset(5.5),
            'balance': 0.1,
            'local_n1': 10,
            'local_n2': 10,
        },
        # 14
        {
            'dim_refset': dim_refset(10.5),
            'balance': 0.3,
            'local_n1': 20,
            'local_n2': 20,
        },
        # 15
        {
            'dim_refset': dim_refset(15.5),
            'balance': 0.2,
            'local_n1': 1000,
            'local_n2': 1000,
        },
        # 16
        {
            'dim_refset': dim_refset(12.5),
            'balance': 0.2,
            'local_n1': 10,
            'local_n2': 10,
        },
        # 17
        {
            'dim_refset': dim_refset(8),
            'balance': 0.75,
            'local_n1': 15,
            'local_n2': 15,
        },
        # 18
        {
            'dim_refset': dim_refset(5.5),
            'balance': 0.75,
            'local_n1': 1000,
            'local_n2': 1000,
        },
        # 19
        {
            'dim_refset': dim_refset(2.2),
            'balance': 1.0,
            'local_n1': 2,
            'local_n2': 2,
        },
        # 20
        {
            'dim_refset': dim_refset(1),
            'balance': 1.0,
            'local_n1': 1,
            'local_n2': 1,
        },
    ]

    # Set local optimizer
    for cur_settings in settings:
        if local_optimizer is True:
            cur_settings['local_optimizer'] = SacessFidesFactory()
        elif local_optimizer is not False:
            cur_settings['local_optimizer'] = local_optimizer

    return [
        settings[0],
        *(itertools.islice(itertools.cycle(settings[1:]), num_workers - 1)),
    ]


class SacessFidesFactory:
    """Factory for :class:`FidesOptimizer` instances for use with :class:`SacessOptimizer`.

    :meth:`__call__` will forward the walltime limit and function evaluation
    limit imposed on :class:`SacessOptimizer` to :class:`FidesOptimizer`.
    Besides that, default options are used.


    Parameters
    ----------
    fides_options:
        Options for the :class:`FidesOptimizer`.
        See :class:`fides.constants.Options`.
    fides_kwargs:
        Keyword arguments for the :class:`FidesOptimizer`. See
        :meth:`FidesOptimizer.__init__`. Must not include ``options``.

    """

    def __init__(
        self,
        fides_options: Optional[dict[str, Any]] = None,
        fides_kwargs: Optional[dict[str, Any]] = None,
    ):
        if fides_options is None:
            fides_options = {}
        if fides_kwargs is None:
            fides_kwargs = {}

        self._fides_options = fides_options
        self._fides_kwargs = fides_kwargs

        # Check if fides is installed
        try:
            import fides  # noqa F401
        except ImportError:
            from ..optimizer import OptimizerImportError

            raise OptimizerImportError("fides")

    def __call__(
        self, max_walltime_s: int, max_eval: int
    ) -> "pypesto.optimize.FidesOptimizer":
        """Create a :class:`FidesOptimizer` instance."""

        from fides.constants import Options as FidesOptions

        options = self._fides_options.copy()
        options[FidesOptions.MAXTIME] = max_walltime_s

        # only accepts int
        if np.isfinite(max_eval):
            options[FidesOptions.MAXITER] = int(max_eval)
        return pypesto.optimize.FidesOptimizer(
            **self._fides_kwargs, options=options
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(fides_options={self._fides_options}, fides_kwargs={self._fides_kwargs})"

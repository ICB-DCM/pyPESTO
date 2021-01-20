"""
Utilities for sampling: apply jitter to a starting point;
resumable and checkpointable sampler
"""

import sys
import time
import os
import shutil
import pickle
from typing import Union, List, Optional, Any, Dict

import numpy as np

from ...problem import Problem
from .model import pypesto_varnames

import arviz

from pymc3 import Model, Point
from pymc3.model import FreeRV, modelcontext
# from pymc3.backends import HDF5
from .hdf5 import HDF5  # Bugfixes
from pymc3.backends.base import MultiTrace
from pymc3.sampling import _choose_backend
from pymc3.util import update_start_vals

# Logging
import warnings
from pymc3.backends.report import logger as pymc3_log
from pymc3.backends.report import _LEVELS as PYMC3_LOG_LEVELS


# This function is extracted from pymc3.sample
def init_random_seed(random_seed: Union[int, None, List[int]], chains: int = 1):
    if random_seed == -1:
        random_seed = None
    if chains == 1 and isinstance(random_seed, int):
        random_seed = [random_seed]
    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]
    return random_seed


# This function is extracted (slighly modified) from pymc3.init_nuts
def jitter(model: Model, start: Union[dict, None], chains: int,
                  *, jitter_first: bool = False):
    if start is None:
        start = model.test_point
    starts = []
    for _ in range(chains):
        new_start = {var: val.copy() for var, val in start.items()}
        for val in new_start.values():
            val[...] += 2 * np.random.rand(*val.shape) - 1
        starts.append(new_start)
    if not jitter_first:
        starts[0] = start
    return starts


class ResumablePymc3Sampler:
    def __init__(self, model: Optional[Model] = None,
                 start: Optional[Dict[str, Any]] = None, *,
                 step: Optional = None,
                 backend: Optional = None,
                 overwrite: bool = False,
                 problem: Optional[Problem] = None,
                 outvars: Optional[List[Union[str, FreeRV]]] = None,
                 tune: Optional[int] = None,
                 progressbar: bool = True,
                 keep_hdf5_open: bool = True,
                 **step_kwargs):

        # Get model from context
        model = modelcontext(model)

        # Create step method
        if step is None:
            step = {}
        if isinstance(step, dict):
            step.default('init', 'adapt_diag')
            step, _start = create_step_method(model, **step)
            assert isinstance(_start, list) and len(_start) == 1
            _start = _start[0]
        else:
            _start = {}

        # Are we going to replace the proposed starting point?
        if start is None:
            assert isinstance(_start, dict)
            start = _start
        elif not isinstance(start, dict):
            raise TypeError('starting point start must be a dictionary')

        if outvars is not None:
            # Convert variables given by name
            if not isinstance(outvars, (list, set)):
                outvars = [outvars]
            outvars = set(outvars)
            outvars = ResumablePymc3Sampler._vars_from_varnames(model, outvars)

        elif problem is not None:
            # Auto-determine variables from pyPESTO problem
            varnames = pypesto_varnames(model, problem)
            outvars = ResumablePymc3Sampler._vars_from_varnames(model, varnames)

        # Create backend from filename
        chain = 0
        if isinstance(backend, str):
            if not(backend.endswith('.h5') or backend.endswith('.hdf5')):
                backend += '.h5'
            if os.path.exists(backend):
                if overwrite:
                    os.remove(backend)
                else:
                    raise FileExistsError(f'HDF5 file {backend} already exists.')
            strace = HDF5(backend, model=model, vars=outvars)
        else:
            strace = _choose_backend(backend, chain, model=model, vars=outvars)

        # Update starting point with test values
        update_start_vals(start, model.test_point, model)

        # Check tune argument
        if tune is None:
            tune = 0
        elif tune < 0:
            raise ValueError("Argument `tune` must be non-negative.")

        # Check progressbar support
        if progressbar:
            try:
                import tqdm
            except ImportError:
                progressbar = False

        # Save to fields
        self._model = model
        self._varnames = [var.name for var in outvars]
        self._start = start
        self._step = step
        self._tune = tune
        self._strace = strace
        self._progressbar = bool(progressbar)
        self._keep_hdf5_open = bool(keep_hdf5_open)
        self._sampling_time = 0.0  # must be written to the report
                                   # in order for arviz conversion to succeed
        self._cur_point = Point(start, model=self.model)
        # NB Point just filters unnecessary keys and returns a dictionary

        # Ensure that the trace is in a loadable state:
        # if we save before drawing even a sample,
        # we need the sampler_vars to be saved in the trace
        self._setup_trace()

    @staticmethod
    def _vars_from_varnames(model, varnames):
        vars = []
        for varname in varnames:
            if isinstance(varname, str):
                found = False
                for var in model.unobserved_RVs:
                    if var.name == varname:
                        vars.append(var)
                        found = True
                        break
                if not found:
                    raise ValueError("could not find a variable named "
                                     f"{varname} among the model's "
                                     "unobserved RVs")
            else:
                vars.append(varname)
        return vars

    @property
    def model(self):
        return self._model

    @property
    def trace(self):
        if self._strace.draws != self._strace.draw_idx:
            raise Exception('invalid trace state (not trimmed)')
        trace = MultiTrace([self._strace])
        report = trace.report
        report._n_draws = self.num_draws
        report._n_tune = self.num_tune
        report._t_sampling = self._sampling_time
        return trace

    @property
    def start(self):
        return self._start

    @property
    def num_samples(self):
        return len(self._strace)

    @property
    def target_tune(self):
        return self._tune

    @property
    def num_draws(self):
        return max(self.num_samples - self.target_tune, 0)

    @property
    def num_tune(self):
        return min(self.target_tune, self.num_samples)

    @property
    def cur_point(self):
        if self._cur_point is None:
            raise Exception('chain was interrupted by an Exception')
        return self._cur_point
        # NB we cannot use self.trace.point(-1),
        #    because we may be saving only a subset of the variables

    @property
    def is_tuning(self):
        return self.target_tune - self.num_samples >= 0

    def increase_tuning_samples(self, samples: int):
        if samples < 0:
            raise ValueError("Argument `samples` must be non-negative.")
        if self.num_samples > self.target_tune:
            raise Exception('.sample(draws) has already been called: '
                            'tuning is no longer possible.')
        self._tune += samples

    def _init_step(self, tune: bool):
        step = self._step
        if self.num_samples == 0:
            step.tune = bool(tune)
            if hasattr(step, 'reset_tuning'):
                step.reset_tuning()
            if hasattr(step, "iter_count"):
                step.iter_count = 0
        if not tune:
            step.stop_tuning()

    def tune(self, min_samples: Optional[int] = None, *,
             quiet: bool = False, warn: bool = True):

        tuning_samples_left = self.target_tune - self.num_samples

        if min_samples is not None:
            if tuning_samples_left < min_samples:
                self.increase_tuning_samples(min_samples - tuning_samples_left)
                assert self.target_tune - self.num_samples == min_samples
                tuning_samples_left = min_samples

        if tuning_samples_left > 0:
            self._init_step(True)
            self._sample(tuning_samples_left, True, warn, False)

        elif tuning_samples_left == 0:
            if not quiet:
                print('WARNING: tuning already completed. '
                      'If additional tuning is required, '
                      'call .increase_tuning_samples(samples) '
                      'before the fist call to .sample(draws)', file=sys.stderr)

        elif not quiet:
            raise Exception('Tuning already completed. '
                            'Since .sample(draws) has already been called, '
                            'no additional tuning is possible')

    def sample(self, draws: int, *, warn: bool = True, rethrow: bool = False):
        if draws < 1:
            raise ValueError("Argument `draws` must be greater than 0.")
        self.tune(quiet=True)
        assert self.target_tune == self.num_tune
        self._init_step(False)
        self._sample(draws, False, warn, rethrow)

    def _sample(self, draws: int, tuning: bool, warn: bool, rethrow: bool):
        if self._keep_hdf5_open and isinstance(self._strace, HDF5):
            with self._strace.activate_file:
                self.__sample(draws, tuning, warn, rethrow)
        else:
            self.__sample(draws, tuning, warn, rethrow)

    def _setup_trace(self, draws=0, chain=0):
        step, strace = self._step, self._strace
        if step.generates_stats and strace.supports_sampler_stats:
            strace.setup(draws, chain, step.stats_dtypes)
        else:
            strace.setup(draws, chain)

    def __sample(self, draws: int, tuning: bool, warn: bool, rethrow: bool):
        assert draws >= 1
        point = self._cur_point
        step, strace = self._step, self._strace
        divergences = 0

        # Allocate space for the new samples
        self._setup_trace(draws)

        iterator = range(draws)
        if self._progressbar:
            from tqdm import tqdm
            iterator = tqdm(iterator)

        t0 = time.perf_counter()
        try:
            for i in iterator:
                if step.generates_stats:
                    point, stats = step.step(point)
                    if strace.supports_sampler_stats:
                        strace.record(point, stats)
                        if not step.tune and stats and stats[0].get("diverging"):
                            divergences += 1
                    else:
                        strace.record(point)
                else:
                    point = step.step(point)
                    strace.record(point)

                if self._progressbar:
                    if tuning:
                        descr = 'Tuning'
                    else:
                        descr = f'Sampling ({divergences} divergences)'
                    iterator.set_description(descr)

        # If by any chance we interrupted before ending,
        # trim the trace removing the unsampled samples by calling .close()
        # TODO how to handle warnings for a resumable chain?
        #      Problems:
        #        * they are not saved to disk
        #        * warnings like non-convergence do not refer to
        #          specific samples: can we allow more than one per chain?
        #        * if we add multiple times warnings to the strace,
        #          will we get many duplicates?
        except:
            self._cur_point = None  # invalidate current point
            if rethrow:
                raise
        else:
            assert strace.draw_idx == strace.draws
            self._cur_point = point
        finally:
            strace.close()  # if no error occured this should be a no-op
            self._sampling_time += time.perf_counter() - t0
            sys.stdout.flush() # Flush progress bar output
            sys.stderr.flush()
            if warn:
                self.log_sampler_warnings()

    def log_sampler_warnings(self):
        if hasattr(self._step, "warnings"):
            for warn in self._step.warnings():
                level = PYMC3_LOG_LEVELS[warn.level]
                pymc3_log.log(level, warn.message)

    # NB since at the moment we only run one chain,
    #    the standard PyMC3 convergence checks are meaningless
    # def log_convergence_checks(self):
    #     if self.num_draws < 100:
    #         warnings.warn(
    #             "The number of samples is too small "
    #             "to check convergence reliably."
    #         )
    #     else:
    #         trace = self.trace  # NB self.trace is generated on demand
    #         idata = arviz.from_pymc3(
    #             trace, model=self.model, save_warmup=False
    #         )
    #         trace.report._run_convergence_checks(idata, self.model)
    #         trace.report._log_summary()

    @staticmethod
    def load_HDF5(path, model, outvars, chain):
        strace = HDF5(path, model, outvars)
        strace.chain = chain  # needed to get the sampler_vars
        sampler_vars = strace.sampler_vars
        strace.setup(0, chain, sampler_vars)  # setup with zero additional draws
        return strace

    def save(self, path: str):
        state = self.__dict__.copy()
        if isinstance(self._strace, HDF5):
            state['_strace'] = self._strace.name
        else:
            raise NotImplementedError('pickling implemented '
                                      'only for HDF5 backend')
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str):
        return cls._load(path)

    @classmethod
    def _load(cls, path: str, datapath: Optional[str] = None):
        self = object.__new__(cls)
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        outvars = ResumablePymc3Sampler._vars_from_varnames(
            self.model, self._varnames
        )
        if datapath is None:
            datapath = self._strace
        self._strace = ResumablePymc3Sampler.load_HDF5(
            datapath, self.model, outvars, chain=0
        )
        return self


class CheckpointablePymc3Sampler:
    def __init__(self, folder: str, *args, start_branch: str = 'root',
                 backend: str = 'hdf5', overwrite: bool = False, **kwargs):
        if backend.lower() != 'hdf5':
            raise NotImplementedError('Only the HDF5 backend is implemented.')
        if os.path.exists(folder):
            if overwrite:
                shutil.rmtree(folder)
            else:
                raise FileExistsError(f'path {folder} already exists!')
        self._folder = os.path.abspath(folder)
        self._cur_branch_file = os.path.join(self.folder, 'current_branch.txt')
        self._cur_branch = start_branch
        self._make_branch_folder(start_branch)
        self._sampler = ResumablePymc3Sampler(
            *args,
            backend=self.branch_trace(start_branch),
            **kwargs
        )
        with open(self._cur_branch_file, 'w') as f:
            f.write(start_branch)
        self.flush()

    @property
    def folder(self):
        return self._folder

    @property
    def branches(self):
        return set([
            name for name in os.listdir(self.folder)
            if os.path.isdir(os.path.join(self.folder, name)) \
               and not name.startswith('.')
        ])

    @property
    def cur_branch(self):
        return self._cur_branch

    def branch_folder(self, branch):
        return os.path.join(self.folder, branch)

    def branch_trace(self, branch):
        return os.path.join(self.branch_folder(branch), 'trace.h5')

    def branch_pickle(self, branch):
        return os.path.join(self.branch_folder(branch), 'sampler.pickle')

    def _make_branch_folder(self, branch):
        folder = self.branch_folder(branch)
        os.makedirs(folder)
        return folder

    def flush(self):
        self._sampler.save(self.branch_pickle(self.cur_branch))

    def load_branch(self, branch):
        if self.cur_branch != branch:
            self._load_branch(branch)

    def _load_branch(self, branch, flush: bool = True, write: bool = True):
        if flush:
            self.flush()

        if not os.path.exists(self.branch_folder(branch)):
            raise FileNotFoundError(f'branch {branch} does not exist')

        if not os.path.exists(self.branch_pickle(branch)):
            raise FileNotFoundError(f'pickle file for branch {branch} '
                                    'does not exist')

        path = self.branch_pickle(branch)
        datapath = self.branch_trace(branch)
        self._sampler = ResumablePymc3Sampler._load(path, datapath)
        self._cur_branch = branch

        if write:
            with open(self._cur_branch_file, 'w') as f:
                f.write(branch)

    def fork_branch(self, new_branch: str, src_branch: str = None, *,
                    load: bool = False):
        if src_branch is None:
            src_branch = self.cur_branch
        if src_branch == self.cur_branch:
            self.flush()

        src_folder = self.branch_folder(src_branch)
        dst_folder = self.branch_folder(new_branch)

        if not os.path.exists(src_folder):
            raise FileNotFoundError(f'branch {src_branch} does not exist')
        if os.path.exists(dst_folder):
            raise FileExistsError(f'branch {new_branch} already exists')

        shutil.copytree(src_folder, dst_folder)

        if load:
            self.load_branch(new_branch)

    def delete_branch(self, branch: Optional[str] = None, *,
                      verbose: bool = True):
        if branch is None:
            branch = self.cur_branch
        # If deleting the current branch, switch to a different one
        if branch == self.cur_branch:
            candidates = self.branches.copy()
            candidates.remove(branch)
            if len(candidates) == 0:
                raise Exception('cannot remove the only branch present')
            new_branch = candidates.pop()
            if verbose:
                print('WARNING: deleting current branch; '
                      f'switching to {new_branch} branch', file=sys.stderr)
            self._load_branch(new_branch, flush=False)  # no need to flush
        # Delete the branch
        shutil.rmtree(self.branch_folder(branch))

    @classmethod
    def load(cls, folder: str, branch: Optional[str] = None):
        if not os.path.exists(folder):
            raise FileNotFoundError(f'folder {folder} does not exist')
        cur_branch_file = os.path.join(folder, 'current_branch.txt')
        if not os.path.exists(cur_branch_file):
            raise FileNotFoundError('current branch file not found '
                                    f'inside folder {folder}')
        if branch is None:
            with open(cur_branch_file, 'r') as f:
                branch = f.read().strip()
        self = object.__new__(cls)
        self._folder = os.path.abspath(folder)
        self._cur_branch_file = os.path.abspath(cur_branch_file)
        self._load_branch(branch, flush=False, write=False)
        return self

    @staticmethod
    def load_single_branch(folder: str, branch: str):
        if not os.path.exists(folder):
            raise FileNotFoundError(f'folder {folder} does not exist')
        branch_folder = os.path.join(folder, branch)
        if not os.path.exists(branch_folder):
            raise FileNotFoundError(f'branch {branch} does not exist')
        path = os.path.join(branch_folder, 'sampler.pickle')
        datapath = os.path.join(branch_folder, 'trace.h5')
        if not os.path.exists(path) or not os.path.exists(datapath):
            raise FileNotFoundError(f'sampler data for branch {folder} does not exist')
        return ResumablePymc3Sampler._load(path, datapath)

    @property
    def model(self):
        return self._sampler._model

    @property
    def trace(self):
        return self._sampler.trace

    @property
    def start(self):
        return self._sampler._start

    @property
    def num_samples(self):
        return self._sampler.num_samples

    @property
    def target_tune(self):
        return self._sampler.target_tune

    @property
    def num_draws(self):
        return self._sampler.num_draws

    @property
    def num_tune(self):
        return self._sampler.num_tune

    @property
    def is_tuning(self):
        return self._sampler.is_tuning

    @property
    def cur_point(self):
        return self._sampler.cur_point

    def increase_tuning_samples(self, samples: int):
        try:
            return self._sampler.increase_tuning_samples(samples)
        finally:
            self.flush()

    def tune(self, min_samples: Optional[int] = None, *,
             quiet: bool = False, warn: bool = False):
        try:
            return self._sampler.tune(min_samples, quiet=quiet, warn=warn)
        finally:
            self.flush()

    def sample(self, draws: int, *, warn: bool = False):
        try:
            return self._sampler.sample(draws, warn=warn)
        finally:
            self.flush()

    def log_sampler_warnings(self):
        return self._sampler.log_sampler_warnings()

    # def log_convergence_checks(self):
    #     return self._sampler.log_convergence_checks()

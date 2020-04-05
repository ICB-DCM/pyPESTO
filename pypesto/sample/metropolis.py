import numpy as np
from typing import Any, Callable, Dict, Union, Tuple
import time

from ..objective import Objective
from ..problem import Problem
from ..objective import History
from .sampler import Sampler, InternalSampler
from .result import McmcPtResult


class MetropolisSampler(Sampler, InternalSampler):
    """Simple Metropolis-Hastings sampler with fixed proposal variance."""

    #def __init__(self, options: Dict = None):
    #    self.options = MetropolisSampler.translate_options(options)

    @staticmethod
    def default_options() -> Dict:
        '''
        The default values of all sampler options are specified here.

        Returns
        -------
        A dictionary of options.
        '''
        defaults = {
            'n_samples': 1000,
        }
        return defaults

    #@staticmethod
    #def translate_options(options):
    #    default_options = {
    #        'n_samples': 1000
    #    }
    #    if options is None:
    #        options = {}
    #    for key, val in options:
    #        if key not in default_options:
    #            raise KeyError(f"Cannot handle key {key}.")
    #        default_options[key] = val
    #    return default_options

    def sample(
            self, problem: Problem, x0: np.ndarray = None
    ) -> Union[McmcPtResult, Any]:

        # set up objective history
        objective = problem.objective
        objective.history = History()

        start_time = time.time()

        x = x0
        llh = - objective(x)
        trace_x = [x]
        trace_fval = [-llh]

        for _ in range(self.options['n_samples']-1):
            x, llh = self.perform_step(x, llh, objective)

            trace_x.append(x)
            trace_fval.append(-llh)

        result = McmcPtResult(
            trace_x=np.array([trace_x]),
            trace_fval=np.array([trace_fval]),
            temperatures=np.array([1.]),
            time=time.time()-start_time,
            n_fval=objective.history.n_fval,
            n_grad=objective.history.n_grad,
            n_hess=objective.history.n_hess
        )

        return result

    def perform_step(self, x: np.ndarray, llh: float, objective: Objective):
        x_new = x + 0.5 * np.random.randn(len(x))
        llh_new = - objective(x_new)

        u = np.random.uniform(0, 1)
        if np.log(u) < llh_new - llh:
            x = x_new
            llh = llh_new

        return x, llh

from .samplers.AdaptiveMetropolis import adaptive_metropolis_sampler_methods
class AdaptiveMetropolisSampler(Sampler, InternalSampler):
    @staticmethod
    def default_options() -> Dict:
        '''
        The default values of all sampler options are specified here.

        Returns
        -------
        A dictionary of options.
        '''
        defaults = {
            'n_samples': 1000,
            #Adaptive Metropolis settings
            'debug': False,
            #'lower_bounds': lower_bounds,
            #'upper_bounds': upper_bounds,
            'decay_constant': 0.51,
            'threshold_sample': 1,
            'regularization_factor': 1e-6,
            #'log_posterior_callable': logpdf,
        }
        return defaults

    def sample(
            self,
            problem: Problem,
            x0: np.ndarray,
            cov0: np.ndarray = None
    ) -> McmcPtResult:
        #self.additional_samples(n_samples)
        #state = self.state
        #state.chain = self.get_chain()

        start_time = time.time()

        if cov0 is None:
           cov0 = np.eye(len(x0))

        x = x0
        cov = adaptive_metropolis_sampler_methods.regularize_covariance(
            cov0,
            self.options['regularization_factor'],
            len(x0),
            MAGIC_DIVIDING_NUMBER = 1000
        )
        objective = problem.objective
        objective.history = History()
        llh = - objective(x)
        x_bar = x
        cov_bar = cov
        cov_scalf = 1 #rewrite as default option?

        trace_x = [x]
        trace_fval = [-llh]
        # uncomment when implementing debug code
        #trace_mean = [x]
        #trace_cov = [cov]

        for n_x in range(self.options['n_samples']-1):
            # if debug, add accepted bool to output list here
            x, llh, cov, x_bar, cov_bar, cov_scalf = self.perform_step(
                x,
                llh,
                objective,
                cov,
                problem,
                x_bar,
                cov_bar,
                cov_scalf,
                n_x
            )
            trace_x.append(x)
            trace_fval.append(-llh)
            # uncomment when implementing debug code
            #n_accepted += accepted
            #trace_acceptance.append(100*n_accepted/(n_x+1)) #not same kind of trace as other trace_ lists...
            #trace_cov_scalf.append(cov_scalf)
            #trace_cov_bar.append(cov_bar)

        result = McmcPtResult(
            trace_x=np.array([trace_x]),
            trace_fval=np.array([trace_fval]),
            temperatures=np.array([1.]),
            time=time.time()-start_time,
            n_fval=objective.history.n_fval,
            n_grad=objective.history.n_grad,
            n_hess=objective.history.n_hess
        )

        return result

    def perform_step(
            self,
            x0: np.ndarray,
            llh0: float,
            objective: Objective,
            cov0: np.ndarray,
            problem: Problem,
            x_bar0: np.ndarray,
            cov_bar0: np.ndarray,
            cov_scalf0: float,
            n_x,
            beta: float = 1
    ):
        x_result = adaptive_metropolis_sampler_methods.try_sampling(
            objective,
            x0,
            llh0,
            cov0,
            problem.lb,
            problem.ub,
            self.options['debug'],
            beta
        )

        x = x_result['sample']
        llh = x_result['log_posterior']

        cov_result = adaptive_metropolis_sampler_methods.estimate_covariance(
            x_bar0,
            cov_bar0,
            x,
            self.options['threshold_sample'],
            self.options['decay_constant'],
            cov_scalf0,
            x_result['log_acceptance'],
            self.options['regularization_factor'],
            len(x),
            n_x
        )

        x_bar = cov_result['historical_mean']
        cov_bar = cov_result['historical_covariance']
        cov_scalf = cov_result['covariance_scaling_factor']
        cov = cov_result['covariance']

        return x, llh, cov, x_bar, cov_bar, cov_scalf


"""
Utilities for converting pymc3.MultiTrace results to
arviz.InferenceData and pypesto.McmcPtResult.
"""

from typing import List, Optional, Union

import numpy as np

import pymc3 as pm
import arviz as az
from arviz import InferenceData

from ...problem import Problem
from ...result import Result
from ..result import McmcPtResult
from .model import pymc3_vector_parname
from .theano import CachedObjective


def pymc3_to_arviz(model: pm.Model, trace: pm.backends.base.MultiTrace, *,
                   problem: Problem = None, x_names: List[str] = None,
                   save_warmup: bool = True, **kwargs):
    if problem is not None and x_names is not None:
        raise ValueError('the problem and x_names keyword arguments '
                         'cannot both be given.')

    # Determine if free parameters were vectorized
    if len(model.free_RVs) == 1:
        theta = model.free_RVs[0]
        if len(theta.distribution.shape) > 0:
            # theta is a tensor variable
            assert len(theta.distribution.shape) == 1  # should only be vector
            if problem is None and x_names is None:
                raise ValueError('if vectorize is True, one of the problem '
                                 'or x_names keyword arguments must be given.')
            if x_names is None:
                x_names = [problem.x_names[i] for i in problem.x_free_indices]
            kwargs['coords'] = {"free_parameter": x_names}
            kwargs['dims'] = {pymc3_vector_parname(x_names): ["free_parameter"]}

    # NB when the observed data is a random variable,
    #    the standard conversion to arviz by arviz.from_pymc3
    #    results in an InferenceData object that cannot be saved to disk
    #    (can be worked with by saving the trace
    #     and recreating the InferenceData each time,
    #     but probably not what the user expects)
    # TODO try to eliminate the need for the log_post variable
    #      it should be easy in the vectorized case
    return CustomPyMC3Converter(
        trace=trace,
        model=model,
        log_likelihood=True,  # pymc3 log-likelihood == negative obj. value
                              #                      == real model's posterior
        save_warmup=save_warmup,
        **kwargs
    ).to_inference_data()


class CustomPyMC3Converter(az.data.io_pymc3.PyMC3Converter):
    # NB if we want to save only the untrasformed variables,
    #    we cannot rely on PyMC3 to compute the objective function
    def _extract_log_likelihood(self, trace):
        if self.log_likelihood is not True:
            raise ValueError('expected log_likelihood to be True!')
        if self.model is None:
            raise ValueError('expected model not to be None!')
        if trace is None:
            raise ValueError('expected trace not to be None!')
        if len(self.model.observed_RVs) != 1:
            raise ValueError('expected number of observed RVs to be one!')
        # Get pyPESTO objective
        var = self.model.observed_RVs[0]
        obj = var.distribution.logp._objective
        if isinstance(obj, CachedObjective):
            obj = obj.objective
        # Create trace for log-likelihoods
        try:
            log_likelihood_dict = self.pymc3.sampling._DefaultTrace(  # pylint: disable=protected-access
                len(trace.chains)
            )
        except AttributeError:
            raise AttributeError(
                "Installed version of ArviZ requires PyMC3>=3.8. Please upgrade with "
                "`pip install pymc3>=3.8` or `conda install -c conda-forge pymc3>=3.8`."
            )
        # Determine if parameters are stored as a vector or separately
        # and create a function mapping a point from a PyMC3 chain
        # to a pyPESTO vector of parameters
        to_params = lambda point : [float(point[name]) for name in obj.x_names]
        if len(self.model.free_RVs) == 1:
            theta = self.model.free_RVs[0]
            theta_shape = theta.distribution.shape
            if len(theta_shape) > 0:
                # theta is a tensor variable
                assert len(theta_shape) == 1  # should only be vector
                theta_name = pymc3_vector_parname(obj.x_names)
                # NB it may be that theta_name != theta.name
                #    The reason is that theta may be
                #    a transformed version of the real parameter vector
                to_params = lambda point : point[theta_name]
        # Compute objective values
        # NB PyMC3 log-likelihood == negative obj. value
        #                         == real model's posterior
        for chain in trace.chains:
            log_like_chain = [
                [-obj(to_params(point))] for point in trace.points([chain])
            ]
            log_likelihood_dict.insert(var.name, np.stack(log_like_chain), chain)
        return log_likelihood_dict.trace_dict

    # NB this function is as in arviz, except for one line
    def to_inference_data(self):
        id_dict = {
            "posterior": self.posterior_to_xarray(),
            "sample_stats": self.sample_stats_to_xarray(),
            "log_likelihood": self.log_likelihood_to_xarray(),
            "posterior_predictive": self.posterior_predictive_to_xarray(),
            "predictions": self.predictions_to_xarray(),
            **self.priors_to_xarray(),
            # "observed_data": self.observed_data_to_xarray(), # COMMENTED OUT
        }
        if self.predictions:
            id_dict["predictions_constant_data"] = self.constant_data_to_xarray()
        else:
            id_dict["constant_data"] = self.constant_data_to_xarray()
        return InferenceData(save_warmup=self.save_warmup, **id_dict)


# NB samplers like AdaptiveMetropolisSampler include the starting point
#    in the trace. Since the NUTS sampler may modify the starting point,
#    we cannot include the starting point in this case
def arviz_to_pypesto(problem: Problem, azdata: az.InferenceData,
                     save_warmup: bool = True,
                     burn_in: Union[None, int, str] = 'auto',
                     result: Optional[Result] = None,
                     full: bool = True):

    trace_x, trace_neglogpost = \
        _extract_trace(problem, azdata.posterior, azdata.log_likelihood)

    # Append the warm-up trace, if present and requested to
    if save_warmup and hasattr(azdata, 'warmup_posterior'):
        warmup_trace_x, warmup_trace_neglogpost = \
            _extract_trace(problem, azdata.warmup_posterior,
                           azdata.warmup_log_likelihood)

        if burn_in == 'auto':
            burn_in = warmup_trace_neglogpost.shape[1]

        trace_x = \
            np.concatenate((trace_x, warmup_trace_x), axis=1)
        trace_neglogpost = \
            np.concatenate((trace_neglogpost, warmup_trace_neglogpost), axis=1)
    elif burn_in == 'auto':
        burn_in = None

    # Build pyPESTO sampling result
    mcmc_result = McmcPtResult(
        trace_x=np.array(trace_x),
        trace_neglogpost=np.array(trace_neglogpost),
        trace_neglogprior=np.full(trace_neglogpost.shape, np.nan),
        betas=np.array([1.] * trace_x.shape[0]),
        burn_in=burn_in,
    )

    # Create a full pyPESTO Result
    if full or result is not None:
        if result is None:
            result = Result(problem)
        result.sample_result = mcmc_result
        return result
    else:
        return mcmc_result


def _extract_trace(problem: Problem, posterior, log_likelihood):
    # Parameter values
    trace_x = np.asarray(posterior.to_array())
    if len(trace_x.shape) == 4:
        # vectorized parameters
        # array dimensions are ordered as
        # (variable, chain, draw, variable coordinates)
        assert trace_x.shape[0] == 1  # all free variables have been packed
                                      # in a single vector variable
        trace_x = np.squeeze(trace_x, axis=0)
    else:
        # array dimensions are ordered as
        # (variable, chain, draw)
        assert len(trace_x.shape) == 3
        trace_x = trace_x.transpose((1, 2, 0))

    # Since the priors in the pymc3 model are artificial
    # and since pypesto objective include the real prior,
    # the log-likelihood of the pymc3 model
    # is actually the real model's log-posterior
    # (i.e., the negative objective value)
    # TODO this is only the negative objective values
    trace_neglogpost = np.asarray(log_likelihood.to_array())
    # Remove trailing dimensions
    trace_neglogpost = np.reshape(trace_neglogpost,
                                  trace_neglogpost.shape[1:-1])
    # Flip sign
    trace_neglogpost = - trace_neglogpost

    if trace_x.shape[0] != trace_neglogpost.shape[0] \
            or trace_x.shape[1] != trace_neglogpost.shape[1] \
            or trace_x.shape[2] != problem.dim:
        raise ValueError("Trace dimensions are inconsistent")

    return trace_x, trace_neglogpost

"""
Utilities for creating PyMC3 step methods outside of PyMC3 sampling methods.
"""

import logging

import pymc3 as pm

from pymc3.model import all_continuous
from pymc3.step_methods import CompoundStep
from pymc3.sampling import assign_step_methods, init_nuts

from theano.gradient import NullTypeGradError


pymc3_log = logging.getLogger("pymc3")


# Looking at pymc3.sample,
# determine which arguments can be forwarded to init_nuts
def filter_create_step_method_kwargs(kwargs):
    keys_to_delete = (
        'draws',
        'start',
        'trace',
        'chain_idx',
        'cores',
        'tune',
        'discard_tuned_samples',
        'compute_convergence_checks',
        'callback',
        'return_inferencedata',
        'idata_kwargs'
    )
    return { key : value
             for (key, value) in kwargs.items()
             if key not in keys_to_delete }


def remove_init_nuts_specific_kwargs(kwargs):
    keys_to_delete = (
        'init',
        'chains',
        'n_init',
        'random_seed',
        'progressbar'
    )
    return { key : value
             for (key, value) in kwargs.items()
             if key not in keys_to_delete }


# This function is extracted from pymc3.sample
def create_step_method(model, step=None, init='auto', **kwargs):
    start = None
    step_kwargs = remove_init_nuts_specific_kwargs(kwargs)

    if step is None and init is not None and all_continuous(model.vars):
        try:
            # By default, try to use NUTS
            pymc3_log.info("Auto-assigning NUTS sampler...")
            start, step = init_nuts(init=init, model=model, **kwargs)

        except (AttributeError, NotImplementedError, NullTypeGradError):
            # gradient computation failed
            pymc3_log.info("Initializing NUTS failed. "
                           "Falling back to elementwise auto-assignment.")
            pymc3_log.debug("Exception in init nuts", exec_info=True)
            with model:  # TODO remove when bug fixed in upstream
                step = assign_step_methods(model, step, step_kwargs=step_kwargs)

    else:
        with model:  # TODO remove when bug fixed in upstream
            step = assign_step_methods(model, step, step_kwargs=step_kwargs)

    if isinstance(step, list):
        step = CompoundStep(step)

    return step, start

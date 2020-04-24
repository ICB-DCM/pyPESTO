import scipy.stats
import numpy as np
from typing import Optional, Union, Iterable, Dict, Callable, List, Tuple
import copy
import math

from .constants import FVAL, GRAD

class Priors:
    '''
    Handles parameter prior distributions.

    This includes producing parameter-specific prior distribution functions
    based on specified prior hyperparameters.

    TODO: Rewrite such that prior definitions are a named dictionary, with
          parameter Id's as the keys?
    '''

    def __init__(
            self,
            x_priors_defs: Iterable[Dict]
    ) -> List['Priors']:
        '''

        Arguments
        ---------
        x_priors_defs:
            A list of dictionaries, where each dictionary defines the prior for
            the parameter with the same index in `dim_full` (see the `Problem`
            class description), with prior type-specific keys, and the
            following non-specific keys:
            'type':
                The type of prior. Currently support types are described
                in the `Prior` class.
            Parameters without priors (such as fixed parameters) should have
            `None` at their respective indicies.
        '''
        self.priors_callables = []
        for x_prior_def in x_priors_defs:
            if x_prior_def is None:
                self.priors_callables += [None]
            elif Prior.check_requirements(x_prior_def):
                self.priors_callables += [Prior.get_callables(x_prior_def)]
            else:
                raise Exception('An error occurred while processing required '
                                'prior hyperparameters, for the prior '
                                'definition:\n'
                                f'{x_prior_def}')

    def __call__(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...]
    ) -> Dict:
        '''
        TODO: Check that it actually returns np.ndarray...
        '''
        priors_results = {FVAL: [], GRAD: []}
        if 0 in sensi_orders:
            priors_results[FVAL] = []
        if 1 in sensi_orders:
            priors_results[GRAD] = []
        for index, x_i in enumerate(x):
            if self.priors_callables[index] is not None:
                if 0 in sensi_orders:
                    priors_results[FVAL] += \
                        [self.priors_callables[index][FVAL](x_i)]
                if 1 in sensi_orders:
                    priors_results[GRAD] += \
                        [self.priors_callables[index][GRAD](x_i)]
        return {k: np.array(v) for k, v in priors_results.items()}

class Prior:
    '''
    Describes supported prior distributions.
    Provides methods to
    - ensure priors are specified appropriately,
    - produce parameter-specific prior distributions as callables

    Some of the methods in this class involve identical arguments. These are
    described here.

    Arguments
    ---------
    prior_type:
        The name of the prior distribution. Supports prior distributions
        are indicated in the `_requirements` method.
    x_prior_def:
        A dictionary with keys that should match the requirements for the
        specified prior type.

    TODO: Could replace `scipy` functions with the subset of functions supported
          by `jax` (https://github.com/google/jax), and then use automatic
          differentiation, instead of `scipy.misc.derivative`.
    '''
    @staticmethod
    def _describe(prior_type: str) -> Dict:
        '''
        Returns information about the prior.

        Returns
        -------
        A dictionary with the following keys.
        'requirements':
            The required hyperparameters for the prior distribution.
        'translation':
            A translation of specified hyperparameters into required
            hyperparameters (for example, facilitates the translation of
            `mean` into `loc`). Possibly unnecessary, certainly optional.
        'package':
            The python package that contains the prior distribution. This is
            useful to specify, because package-dependent code can then be used
            to simplify things (for example `scipy.stats` functions generally
            provide `logpdf` functions).
        'callable':
            The generic probability distribution function of the prior, with
            unspecified hyperparameters. Note that the hyperparameters
            in the `'requirements'` entry (after translation) must be named
            arguments to this callable, and the parameter should be an unnamed
            argument.
        #'callable_grad':
        #    A function that takes a parameter value argument, as well as the
        #    required hyperparameters (see `'requirements'`), and returns the
        #    gradient of `callable` at the parameter value.

        Returns the list of required hyperparameters for priors. This and the
        `check_type` methods are possibly unnecessary (useful for debugging).
        '''
        requirements = []
        translations = []
        package = 'custom'
        prior_callable = None
        prior_callable_grad = None
        if prior_type == 'normal':
            requirements = ['loc', 'scale']
            translations = [('mean', 'loc'),
                           ('std', 'scale')]
            package = 'scipy'
            prior_callable = scipy.stats.norm.logpdf
            #prior_callable_grad = lambda x0, loc, scale: scipy.misc.derivative(
            #    lambda x: prior_callable(x, loc, scale), x0
            #)
        elif prior_type == 'lognormal':
            requirements = ['loc', 'scale']
            translations = [('mean', 'loc'),
                           ('std', 'scale')]
            package = 'scipy'
            prior_callable = scipy.stats.lognorm.logpdf
            #prior_callable_grad = lambda x0, loc, scale: scipy.misc.derivative(
            #    lambda x: prior_callable(x, loc, scale), x0
            #)
        else:
            raise NameError(f'"{prior_type}" priors are not supported.')

        return {
            'requirements': requirements,
            'translations': dict(translations),
            'package': package,
            'callable': prior_callable,
            #'callable_grad': prior_callable_grad
        }

    @staticmethod
    def check_requirements(x_prior_def: Dict) -> bool:
        '''
        Returns True if the prior definition contains the required prior
        hyperparameters for the specified prior type, else returns `False`.
        '''
        # TODO: Ensure that, for example, both `mean` and `loc` are not
        # specified in requirements (as `mean` translated to `loc` for scipy
        # distributions)?
        prior_type = x_prior_def['type']
        prior_description = Prior._describe(prior_type)
        #requirements = copy.copy(prior_description['requirements'])
        #requirements = copy.copy(prior_description['requirements'])
        #translations = copy.copy(Prior._describe(prior_type['translations'])
        for translation in prior_description['translations']:
            if translation[0] in x_prior_def['hyperparameters']:
                prior_description['requirements'][
                    prior_description['requirements'].index(translation[1])] = \
                    translation[0]
        return all([r in x_prior_def['hyperparameters']
                    for r in prior_description['requirements']])
        #return all([r in x_prior_def
        #            for r in Prior._describe(prior_type)['requirements'] + \
        #            list(dict(
        #                Prior._describe(prior_type)['translations']
        #            ).keys())])

    @staticmethod
    def _translate_requirements(prior_description: Dict,
                               x_prior_def: Dict) -> Dict:
        translated = {}
        #prior_description = self._describe(prior_type)
        translations = dict(prior_description['translations'])
        requirements = prior_description['requirements']
        for hyperparameter, value in x_prior_def['hyperparameters'].items():
            if hyperparameter in translations:
                translated[translations[hyperparameter]] = value
            elif hyperparameter in requirements:
                translated[hyperparameter] = value
            else:
                # If this occurs, possible cause is described in the TODO
                # comment in the `check_requirements` method.
                raise NameError('Unrecognised prior distribution '
                                f'hyperparameter: {hyperparameter}, for prior '
                                f'of type {x_prior_def["type"]}')
        return translated

    @staticmethod
    def _scale(scale: str, x: float) -> float:
        if scale == 'lin' or scale == None:
            return x
        elif scale == 'log':
            return math.exp(x)
        elif scale == 'log10':
            return 10**x
        else:
            raise NameError(f'Unrecognised scale: {scale}.')

    @staticmethod
    def get_callables(x_prior_def: Dict) -> Callable:
        '''
        Returns the appropriate prior distribution as a callable, which is
        customised with hyperparameters that are specified in the definition.
        Also returns the derivative function of the callable.

        TODO: Implement sensi_orders here, to only return gradient function if
              required. Would require passing sensi_orders from Problem class??
        '''
        prior_type = x_prior_def['type']
        prior_description = Prior._describe(prior_type)
        # translated hyperparameters
        kwargs = Prior._translate_requirements(prior_description, x_prior_def)

        function = None
        gradient = None
        if prior_description['package'] == 'scipy':
            scale = prior_description['scale'] \
                if 'scale' in prior_description else None
            function = lambda x: prior_description['callable'](
                Prior._scale(scale, x),
                **kwargs
            )
            gradient = lambda x0: scipy.misc.derivative(
                lambda x: callable_fun(Prior._scale(scale, x)),
                Prior._scale(scale, x0)
            )
        return {
            FVAL: function,
            GRAD: gradient
        }

import abc
from typing import Dict, Union

from ..objective import Objective
from .options import SamplerOptions
from .result import SamplerResult

class Sampler(abc.ABC):

    def __init__(self, options: Union[Dict, SamplerOptions]):
        """Constructor.

        Parameters
        ----------
        options:
            Options configuring the sampler run.
        """
        options = SamplerOptions.create_instance(options)
        self.options = self.__class__.translate_options(options)

    @classmethod
    def translate_options(cls, options: SamplerOptions) -> Dict:
        """Translate options to sampler specific options.
        Default: Do nothing.

        Parameters
        ----------
        options:
            The provided options.

        Returns
        -------
        options:
            The translated options.
        """
        return options

    @abc.abstractmethod
    def sample(self, objective: Objective) -> SamplerResult:
        """"Perform the actual sampling.

        Parameters
        ----------
        objective:
            The objective for which to sample.

        Returns
        -------
        sample_result:
            The sampling results in standardized format.
        """

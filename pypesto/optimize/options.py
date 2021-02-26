from typing import Dict, Union


class OptimizeOptions(dict):
    """
    Options for the multistart optimization.

    Parameters
    ----------
    startpoint_resample:
        Flag indicating whether initial points are supposed to be resampled if
        function evaluation fails at the initial point
    allow_failed_starts:
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.
    """

    def __init__(self,
                 startpoint_resample: bool = False,
                 allow_failed_starts: bool = True):
        super().__init__()

        self.startpoint_resample: bool = startpoint_resample
        self.allow_failed_starts: bool = allow_failed_starts

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(
            maybe_options: Union['OptimizeOptions', Dict]
    ) -> 'OptimizeOptions':
        """
        Returns a valid options object.

        Parameters
        ----------

        maybe_options: OptimizeOptions or dict
        """
        if isinstance(maybe_options, OptimizeOptions):
            return maybe_options
        options = OptimizeOptions(**maybe_options)
        return options

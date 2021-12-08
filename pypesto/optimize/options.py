from typing import Dict, Union


class OptimizeOptions(dict):
    """
    Options for the multistart optimization.

    Parameters
    ----------
    allow_failed_starts:
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.
    report_sres:
        Flag indicating whether sres will be stored in the results object.
        Deactivating this option will improve memory consumption for large
        scale problems.
    report_hess:
        Flag indicating whether hess will be stored in the results object.
        Deactivating this option will improve memory consumption for large
        scale problems.
    """

    def __init__(
        self,
        allow_failed_starts: bool = True,
        report_sres: bool = True,
        report_hess: bool = True,
    ):
        super().__init__()

        self.allow_failed_starts: bool = allow_failed_starts
        self.report_sres: bool = report_sres
        self.report_hess: bool = report_hess

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(
        maybe_options: Union['OptimizeOptions', Dict],
    ) -> 'OptimizeOptions':
        """
        Return a valid options object.

        Parameters
        ----------
        maybe_options: OptimizeOptions or dict
        """
        if isinstance(maybe_options, OptimizeOptions):
            return maybe_options
        options = OptimizeOptions(**maybe_options)
        return options

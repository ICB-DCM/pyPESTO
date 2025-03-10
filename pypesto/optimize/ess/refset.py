"""ReferenceSet functionality for scatter search."""

from typing import Any, Optional

import numpy as np

from .function_evaluator import FunctionEvaluator


class RefSet:
    """Scatter search reference set.

    Attributes
    ----------
    dim:
        Reference set size
    evaluator:
        Function evaluator
    x:
        Parameters in the reference set
    fx:
        Function values at the parameters in the reference set
    n_stuck:
        Counts the number of times a refset member did not lead to an
        improvement in the objective (length: ``dim``).
    """

    def __init__(
        self,
        dim: int,
        evaluator: FunctionEvaluator,
        x: Optional[np.ndarray] = None,
        fx: Optional[np.ndarray] = None,
    ):
        """Construct.

        Parameters
        ----------
        dim:
            Reference set size
        evaluator:
            Function evaluator
        x:
            Initial RefSet parameters.
        fx:
            Function values corresponding to entries in x. Must be provided if
            and only if ``x`` is not ``None``.
        """
        if (x is not None and fx is None) or (x is None and fx is not None):
            raise ValueError(
                "Either both or neither of `x` and `fx` should be provided"
            )

        if dim < 3:
            raise ValueError("RefSet dimension has to be at least 3.")
        self.dim = dim
        self.evaluator = evaluator
        # \epsilon in [PenasGon2017]_
        self.proximity_threshold = 1e-3

        if x is None:
            self.x = self.fx = None
        else:
            self.x = x
            self.fx = fx

        self.n_stuck = np.zeros(shape=[dim])
        self.attributes: dict[Any, np.ndarray] = {}

    def __repr__(self):
        fx = (
            f", fx=[{np.min(self.fx)} ... {np.max(self.fx)}]"
            if self.fx is not None and len(self.fx) >= 2
            else ""
        )
        return f"RefSet(dim={self.dim}{fx})"

    def sort(self):
        """Sort RefSet by quality."""
        order = np.argsort(self.fx)
        self.fx = self.fx[order]
        self.x = self.x[order]
        self.n_stuck = self.n_stuck[order]
        for attribute_name, attribute_values in self.attributes.items():
            self.attributes[attribute_name] = attribute_values[order]

    def initialize_random(
        self,
        n_diverse: int,
    ):
        """Create an initial reference set from random parameters.

        Sample ``n_diverse`` random points, populate half of the RefSet using
        the best solutions and fill the rest with random points.
        """
        # sample n_diverse points
        x_diverse, fx_diverse = self.evaluator.multiple_random(n_diverse)
        self.initialize_from_array(x_diverse=x_diverse, fx_diverse=fx_diverse)

    def initialize_from_array(
        self, x_diverse: np.ndarray, fx_diverse: np.ndarray
    ):
        """Create an initial reference set using the provided points.

        Populate half of the RefSet using the best given solutions and fill the
        rest with a random selection from the remaining points.
        """
        if len(x_diverse) != len(fx_diverse):
            raise ValueError(
                "Lengths of `x_diverse` and `fx_diverse` do not match."
            )
        if self.dim > len(x_diverse):
            raise ValueError(
                "Cannot create RefSet with dimension "
                f"{self.dim} from only {len(x_diverse)} points."
            )

        self.fx = np.full(shape=(self.dim,), fill_value=np.inf)
        self.x = np.full(
            shape=(self.dim, self.evaluator.problem.dim), fill_value=np.nan
        )

        # create initial refset with 50% best values
        num_best = int(self.dim / 2)
        order = np.argsort(fx_diverse)
        self.x[:num_best] = x_diverse[order[:num_best]]
        self.fx[:num_best] = fx_diverse[order[:num_best]]

        # ... and 50% random points
        random_idxs = np.random.choice(
            order[num_best:], size=self.dim - num_best, replace=False
        )
        self.x[num_best:] = x_diverse[random_idxs]
        self.fx[num_best:] = fx_diverse[random_idxs]

    def prune_too_close(self):
        """Prune too similar RefSet members.

        Replace a parameter vector if its maximum relative difference to a
        better parameter vector is below the given threshold.

        Assumes RefSet is sorted.
        """
        # Compare [PenasGon2007]
        #  Note that the main text states that distance between the two points
        #  is normalized to the bounds of the search space. However,
        #  Algorithm 1, line 9 normalizes to x_j instead. The accompanying
        #  code does normalize to max(abs(x_i), abs(x_j)).
        # Normalizing to the bounds of the search space seems more reasonable.
        #  Otherwise, for a parameter with bounds [lb, ub],
        #  where (ub-lb)/ub < proximity_threshold, we would never find an
        #  admissible point.
        x = self.x
        ub, lb = self.evaluator.problem.ub, self.evaluator.problem.lb

        def normalize(x):
            """Normalize parameter vector to the bounds of the search space."""
            return (x - lb) / (ub - lb)

        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # check proximity
                # zero-division may occur here
                with np.errstate(divide="ignore", invalid="ignore"):
                    while (
                        np.max(np.abs(normalize(x[i]) - normalize(x[j])))
                        <= self.proximity_threshold
                    ):
                        # too close. replace x_j.
                        x[j], self.fx[j] = self.evaluator.single_random()
                        self.sort()

    def update(self, i: int, x: np.ndarray, fx: float):
        """Update a RefSet entry."""
        self.x[i] = x
        self.fx[i] = fx
        self.n_stuck[i] = 0

    def replace_by_random(self, i: int):
        """Replace the RefSet member with the given index by a random point."""
        self.x[i], self.fx[i] = self.evaluator.single_random()
        self.n_stuck[i] = 0

    def add_attribute(self, name: str, values: np.ndarray):
        """
        Add an attribute array to the refset members.

        An attribute can be any 1D array of the same length as the refset.
        The added array will be sorted together with the refset members.
        """
        if len(values) != self.dim:
            raise ValueError("Attribute length does not match refset length.")
        self.attributes[name] = values

    def resize(self, new_dim: int):
        """
        Resize the refset.

        If the dimension does not change, do nothing.
        If size is decreased, drop entries from the end (i.e., the worst
        values, assuming it is sorted). If size is increased, the new
        entries are filled with randomly sampled parameters and the refset is
        sorted.

        NOTE: Any attributes are just truncated or filled with zeros.
        """
        if new_dim == self.dim:
            return

        if new_dim < self.dim:
            # shrink
            self.fx = self.fx[:new_dim]
            self.x = self.x[:new_dim]
            self.n_stuck = self.n_stuck[:new_dim]
            for attribute_name, attribute_values in self.attributes.items():
                self.attributes[attribute_name] = attribute_values[:new_dim]
            self.dim = new_dim
        else:
            # grow
            new_x, new_fx = self.evaluator.multiple_random(new_dim - self.dim)
            self.fx = np.append(self.fx, new_fx)
            self.x = np.vstack((self.x, new_x))
            self.n_stuck = np.append(
                self.n_stuck, np.zeros(shape=(new_dim - self.dim))
            )
            for attribute_name, attribute_values in self.attributes.items():
                self.attributes[attribute_name] = np.append(
                    attribute_values, np.zeros(shape=(new_dim - self.dim))
                )
            self.dim = new_dim
            self.sort()

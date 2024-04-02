from typing import Union

import numpy as np

from .ensemble import Ensemble, EnsemblePrediction
from .util import get_prediction_dataset


def get_covariance_matrix_parameters(ens: Ensemble) -> np.ndarray:
    """
    Compute the covariance of ensemble parameters.

    Parameters
    ----------
    ens:
        Ensemble object containing a set of parameter vectors

    Returns
    -------
    covariance_matrix:
        covariance matrix of ensemble parameters
    """
    # call lowlevel routine using the parameter ensemble
    return np.cov(ens.x_vectors.transpose())


def get_covariance_matrix_predictions(
    ens: Union[Ensemble, EnsemblePrediction], prediction_index: int = 0
) -> np.ndarray:
    """
    Compute the covariance of ensemble predictions.

    Parameters
    ----------
    ens:
        Ensemble object containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions
    prediction_index:
        index telling which prediction from the list should be analyzed

    Returns
    -------
    covariance_matrix:
        covariance matrix of ensemble predictions
    """
    # extract the an array of predictions from either an Ensemble object or an
    # EnsemblePrediction object
    dataset = get_prediction_dataset(ens, prediction_index)

    # call lowlevel routine using the prediction ensemble
    return np.cov(dataset)


def get_spectral_decomposition_parameters(
    ens: Ensemble,
    normalize: bool = False,
    only_separable_directions: bool = False,
    cutoff_absolute_separable: float = 1e-16,
    cutoff_relative_separable: float = 1e-16,
    only_identifiable_directions: bool = False,
    cutoff_absolute_identifiable: float = 1e-16,
    cutoff_relative_identifiable: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectral decomposition of ensemble parameters.

    Parameters
    ----------
    ens:
        Ensemble object containing a set of parameter vectors
    normalize:
        flag indicating whether the returned Eigenvalues should be normalized
        with respect to the largest Eigenvalue
    only_separable_directions:
        return only separable directions according to
        cutoff_[absolute/relative]_separable
    cutoff_absolute_separable:
        Consider only eigenvalues of the covariance matrix above this cutoff
        (only applied when only_separable_directions is True)
    cutoff_relative_separable:
        Consider only eigenvalues of the covariance matrix above this cutoff,
        when rescaled with the largest eigenvalue
        (only applied when only_separable_directions is True)
    only_identifiable_directions:
        return only identifiable directions according to
        cutoff_[absolute/relative]_identifiable
    cutoff_absolute_identifiable:
        Consider only low eigenvalues of the covariance matrix with inverses
        above of this cutoff
        (only applied when only_identifiable_directions is True)
    cutoff_relative_identifiable:
        Consider only low eigenvalues of the covariance matrix when rescaled
        with the largest eigenvalue with inverses above of this cutoff
        (only applied when only_identifiable_directions is True)

    Returns
    -------
    eigenvalues:
        Eigenvalues of the covariance matrix
    eigenvectors:
        Eigenvectors of the covariance matrix
    """
    # check inputs
    if sum([only_identifiable_directions, only_separable_directions]) >= 2:
        raise AssertionError(
            "Specify either only identifiable or only separable directions."
        )

    covariance = get_covariance_matrix_parameters(ens)
    return get_spectral_decomposition_lowlevel(
        matrix=covariance,
        normalize=normalize,
        only_separable_directions=only_separable_directions,
        cutoff_absolute_separable=cutoff_absolute_separable,
        cutoff_relative_separable=cutoff_relative_separable,
        only_identifiable_directions=only_identifiable_directions,
        cutoff_absolute_identifiable=cutoff_absolute_identifiable,
        cutoff_relative_identifiable=cutoff_relative_identifiable,
    )


def get_spectral_decomposition_predictions(
    ens: Ensemble,
    normalize: bool = False,
    only_separable_directions: bool = False,
    cutoff_absolute_separable: float = 1e-16,
    cutoff_relative_separable: float = 1e-16,
    only_identifiable_directions: bool = False,
    cutoff_absolute_identifiable: float = 1e-16,
    cutoff_relative_identifiable: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectral decomposition of ensemble predictions.

    Parameters
    ----------
    ens:
        Ensemble object containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions
    normalize:
        flag indicating whether the returned Eigenvalues should be normalized
        with respect to the largest Eigenvalue
    only_separable_directions:
        return only separable directions according to
        cutoff_[absolute/relative]_separable
    cutoff_absolute_separable:
        Consider only eigenvalues of the covariance matrix above this cutoff
        (only applied when only_separable_directions is True)
    cutoff_relative_separable:
        Consider only eigenvalues of the covariance matrix above this cutoff,
        when rescaled with the largest eigenvalue
        (only applied when only_separable_directions is True)
    only_identifiable_directions:
        return only identifiable directions according to
        cutoff_[absolute/relative]_identifiable
    cutoff_absolute_identifiable:
        Consider only low eigenvalues of the covariance matrix with inverses
        above of this cutoff
        (only applied when only_identifiable_directions is True)
    cutoff_relative_identifiable:
        Consider only low eigenvalues of the covariance matrix when rescaled
        with the largest eigenvalue with inverses above of this cutoff
        (only applied when only_identifiable_directions is True)

    Returns
    -------
    eigenvalues:
        Eigenvalues of the covariance matrix
    eigenvectors:
        Eigenvectors of the covariance matrix
    """
    covariance = get_covariance_matrix_predictions(ens)
    return get_spectral_decomposition_lowlevel(
        matrix=covariance,
        normalize=normalize,
        only_separable_directions=only_separable_directions,
        cutoff_absolute_separable=cutoff_absolute_separable,
        cutoff_relative_separable=cutoff_relative_separable,
        only_identifiable_directions=only_identifiable_directions,
        cutoff_absolute_identifiable=cutoff_absolute_identifiable,
        cutoff_relative_identifiable=cutoff_relative_identifiable,
    )


def get_spectral_decomposition_lowlevel(
    matrix: np.ndarray,
    normalize: bool = False,
    only_separable_directions: bool = False,
    cutoff_absolute_separable: float = 1e-16,
    cutoff_relative_separable: float = 1e-16,
    only_identifiable_directions: bool = False,
    cutoff_absolute_identifiable: float = 1e-16,
    cutoff_relative_identifiable: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectral decomposition of ensemble parameters or predictions.

    Parameters
    ----------
    matrix:
        symmetric matrix (typically a covariance matrix) of parameters or
        predictions
    normalize:
        flag indicating whether the returned Eigenvalues should be normalized
        with respect to the largest Eigenvalue
    only_separable_directions:
        return only separable directions according to
        cutoff_[absolute/relative]_separable
    cutoff_absolute_separable:
        Consider only eigenvalues of the covariance matrix above this cutoff
        (only applied when only_separable_directions is True)
    cutoff_relative_separable:
        Consider only eigenvalues of the covariance matrix above this cutoff,
        when rescaled with the largest eigenvalue
        (only applied when only_separable_directions is True)
    only_identifiable_directions:
        return only identifiable directions according to
        cutoff_[absolute/relative]_identifiable
    cutoff_absolute_identifiable:
        Consider only low eigenvalues of the covariance matrix with inverses
        above of this cutoff
        (only applied when only_identifiable_directions is True)
    cutoff_relative_identifiable:
        Consider only low eigenvalues of the covariance matrix when rescaled
        with the largest eigenvalue with inverses above of this cutoff
        (only applied when only_identifiable_directions is True)

    Returns
    -------
    eigenvalues:
        Eigenvalues of the covariance matrix
    eigenvectors:
        Eigenvectors of the covariance matrix
    """
    # get the eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # get a normalized version
    rel_eigenvalues = eigenvalues / np.max(eigenvalues)

    # If no filtering is wanted, we can return
    if not only_identifiable_directions and not only_separable_directions:
        # apply normalization
        if normalize:
            eigenvalues = rel_eigenvalues
        return eigenvalues, eigenvectors

    # Separable directions are wanted: an upper pass filtering is needed
    if only_separable_directions:
        if (
            cutoff_absolute_separable is not None
            and cutoff_relative_separable is not None
        ):
            above_cutoff = np.array(
                [
                    i_eig_abs > cutoff_absolute_separable
                    and i_eig_rel > cutoff_relative_separable
                    for i_eig_abs, i_eig_rel in zip(
                        eigenvalues, rel_eigenvalues
                    )
                ]
            )
        elif cutoff_absolute_separable is not None:
            above_cutoff = eigenvalues > cutoff_absolute_separable
        elif cutoff_relative_separable is not None:
            above_cutoff = rel_eigenvalues > cutoff_relative_separable
        else:
            raise Exception(
                "Need a lower cutoff (absolute or relative, "
                "e.g., 1e-16, to compute separable directions."
            )

        # restrict to those above cutoff
        eigenvalues = eigenvalues[above_cutoff]
        eigenvectors = eigenvectors[:, above_cutoff]
        # apply normlization
        if normalize:
            eigenvalues = rel_eigenvalues[above_cutoff]
        return eigenvalues, eigenvectors

    # Identifiable directions are wanted: an filtering of the inverse
    # eigenvalues is needed (upper pass of inverse = lower pass of original)
    if (
        cutoff_absolute_identifiable is not None
        and cutoff_relative_identifiable is not None
    ):
        below_cutoff = np.array(
            [
                1 / i_eig_abs > cutoff_absolute_identifiable
                and 1 / i_eig_rel > cutoff_relative_identifiable
                for i_eig_abs, i_eig_rel in zip(eigenvalues, rel_eigenvalues)
            ]
        )
    elif cutoff_absolute_identifiable is not None:
        below_cutoff = 1 / eigenvalues > cutoff_absolute_identifiable
    elif cutoff_relative_identifiable is not None:
        below_cutoff = 1 / rel_eigenvalues > cutoff_relative_identifiable
    else:
        raise Exception(
            "Need an inverse upper cutoff (absolute or relative, "
            "e.g., 1e-16, to compute identifiable directions."
        )

    # restrict to those below cutoff
    eigenvalues = eigenvalues[below_cutoff]
    eigenvectors = eigenvectors[:, below_cutoff]
    # apply normlization
    if normalize:
        eigenvalues = rel_eigenvalues[below_cutoff]
    return eigenvalues, eigenvectors

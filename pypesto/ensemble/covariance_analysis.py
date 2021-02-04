import numpy as np
from typing import Union

from ..ensemble import Ensemble, EnsemblePrediction
from .utils import get_prediction_dataset


def get_covariance_matrix_parameters(ens: Ensemble):
    """
    Compute the covariance of ensemble parameters.

    Parameters
    ==========
    ens:
        Ensemble object containing a set of parameter vectors

    Returns
    =======
    covariance_matrix:
        covariance matrix of ensemble parameters
    """

    # call lowlevel routine using the parameter ensemble
    return get_covariance_matrix_lowlevel(dataset=ens.x_vectors.transpose())


def get_covariance_matrix_predictions(
        ens: Union[Ensemble, EnsemblePrediction],
        prediction_index: int = 0) -> np.ndarray:
    """
    Compute the covariance of ensemble predictions.

    Parameters
    ==========
    ens:
        Ensemble object containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions

    prediction_index:
        index telling which prediction from the list should be analyzed

    Returns
    =======
    covariance_matrix:
        covariance matrix of ensemble predictions
    """

    # extract the an array of predictions from either an Ensemble object or an
    # EnsemblePrediction object
    dataset = get_prediction_dataset(ens, prediction_index)

    # call lowlevel routine using the prediction ensemble
    return get_covariance_matrix_lowlevel(dataset=dataset)


def get_spectral_decomposition_parameters(
        ens: Ensemble,
        normalize: bool = False,
        lower_absolute_cutoff: float = None,
        lower_relative_cutoff: float = None,
        inv_upper_absolute_cutoff: float = None,
        inv_upper_relative_cutoff: float = None,
        only_identifiable_directions: bool = False,
        only_separable_directions: bool = False):
    """
    Compute the spectral docmposition of ensemble parameters.

    Parameters
    ==========
    ens:
        Ensemble object containing a set of parameter vectors

    normalize:
        flag indicating whether the returned Eigenvalues should be normalized
        with respect to the largest Eigenvalue

    lower_absolute_cutoff:
        Consider only eigenvalues of the covariance matrix above this cutoff
        (only applied when only_separable_directions is True)

    lower_relative_cutoff:
        Consider only eigenvalues of the covariance matrix above this cutoff,
        when rescaled with the largest eigenvalue
        (only applied when only_separable_directions is True)

    inv_upper_absolute_cutoff:
        Consider only low eigenvalues of the covariance matrix with inverses
        above of this cutoff
        (only applied when only_identifiable_directions is True)

    inv_upper_relative_cutoff:
        Consider only low eigenvalues of the covariance matrix when rescaled
        with the largest eigenvalue with inverses above of this cutoff
        (only applied when only_identifiable_directions is True)

    only_identifiable_directions:
        return only identifiable directions according to inv_upper_cutoff

    only_separable_directions:
        return only separable directions according to lower_cutoff

    Returns
    =======
    eigen_values:
        Eigenvalues of the covariance matrix

    eigen_vectors:
        eigenvectors of the covariance matrix
    """
    covariance = get_covariance_matrix_parameters(ens)
    return get_spectral_decomposition_lowlevel(
        covariance, normalize, lower_absolute_cutoff, lower_relative_cutoff,
        inv_upper_absolute_cutoff, inv_upper_relative_cutoff,
        only_identifiable_directions, only_separable_directions)


def get_spectral_decomposition_predictions(
        ens: Ensemble,
        normalize: bool = False,
        lower_absolute_cutoff: float = None,
        lower_relative_cutoff: float = None,
        inv_upper_absolute_cutoff: float = None,
        inv_upper_relative_cutoff: float = None,
        only_identifiable_directions: bool = False,
        only_separable_directions: bool = False):
    """
    Compute the spectral docmposition of ensemble predictions.

    Parameters
    ==========
    ens:
        Ensemble object containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions

    normalize:
        flag indicating whether the returned Eigenvalues should be normalized
        with respect to the largest Eigenvalue

    lower_absolute_cutoff:
        Consider only eigenvalues of the covariance matrix above this cutoff
        (only applied when only_separable_directions is True)

    lower_relative_cutoff:
        Consider only eigenvalues of the covariance matrix above this cutoff,
        when rescaled with the largest eigenvalue
        (only applied when only_separable_directions is True)

    inv_upper_absolute_cutoff:
        Consider only low eigenvalues of the covariance matrix with inverses
        above of this cutoff
        (only applied when only_identifiable_directions is True)

    inv_upper_relative_cutoff:
        Consider only low eigenvalues of the covariance matrix when rescaled
        with the largest eigenvalue with inverses above of this cutoff
        (only applied when only_identifiable_directions is True)

    only_identifiable_directions:
        return only identifiable directions according to inv_upper_cutoff

    only_separable_directions:
        return only separable directions according to lower_cutoff

    Returns
    =======
    eigen_values:
        Eigenvalues of the covariance matrix

    eigen_vectors:
        eigenvectors of the covariance matrix
    """
    covariance = get_covariance_matrix_predictions(ens)
    return get_spectral_decomposition_lowlevel(
        covariance, normalize, lower_absolute_cutoff, lower_relative_cutoff,
        inv_upper_absolute_cutoff, inv_upper_relative_cutoff,
        only_identifiable_directions, only_separable_directions)


def get_covariance_matrix_lowlevel(dataset):
    """
    Compute the covariance of ensemble parameters or predictions.

    Parameters
    ==========
    dataset:
        numpy array of ensemble parameters or predictions

    Returns
    =======
    covariance_matrix:
        covariance matrix of the dataset
    """
    return np.cov(dataset)


def get_spectral_decomposition_lowlevel(
        matrix_to_analyze: np.ndarray,
        normalize: bool = False,
        lower_absolute_cutoff: float = None,
        lower_relative_cutoff: float = None,
        inv_upper_absolute_cutoff: float = None,
        inv_upper_relative_cutoff: float = None,
        only_identifiable_directions: bool = False,
        only_separable_directions: bool = False):
    """
    Compute the spectral docmposition of ensemble parameters or predictions.

    Parameters
    ==========
    matrix_to_analyze:
        symmetric matrix (typically a covariance matrix) of parameters or
        predictions

    normalize:
        flag indicating whether the returned Eigenvalues should be normalized
        with respect to the largest Eigenvalue

    lower_absolute_cutoff:
        Consider only eigenvalues of the covariance matrix above this cutoff
        (only applied when only_separable_directions is True)

    lower_relative_cutoff:
        Consider only eigenvalues of the covariance matrix above this cutoff,
        when rescaled with the largest eigenvalue
        (only applied when only_separable_directions is True)

    inv_upper_absolute_cutoff:
        Consider only low eigenvalues of the covariance matrix with inverses
        above of this cutoff
        (only applied when only_identifiable_directions is True)

    inv_upper_relative_cutoff:
        Consider only low eigenvalues of the covariance matrix when rescaled
        with the largest eigenvalue with inverses above of this cutoff
        (only applied when only_identifiable_directions is True)

    only_identifiable_directions:
        return only identifiable directions according to inv_upper_cutoff

    only_separable_directions:
        return only separable directions according to lower_cutoff

    Returns
    =======
    eigen_values:
        Eigenvalues of the covariance matrix

    eigen_vectors:
        eigenvectors of the covariance matrix
    """

    # get the eigenvalue decomposition
    eigen_vals, eigen_vectors = np.linalg.eigh(matrix_to_analyze)

    # get a normalized version
    rel_eigen_vals = eigen_vals / np.max(eigen_vals)

    # If no filtering is wanted, we can return
    if not only_identifiable_directions and not only_separable_directions:
        # apply normlization
        if normalize:
            eigen_vals = rel_eigen_vals
        return eigen_vals, eigen_vectors

    if only_identifiable_directions and only_identifiable_directions:
        raise Exception('Asking for only identiafiable and only separable '
                        'directions at the same time makes no sense. The '
                        'applied filters are mutually exclusive.')

    # Separable directions are wanted: an upper pass filtering is needed
    if only_separable_directions:
        if lower_absolute_cutoff is not None and \
                lower_relative_cutoff is not None:
            above_cutoff = np.array([
                eigen_vals[i_eig] > lower_absolute_cutoff and
                rel_eigen_vals[i_eig] > lower_relative_cutoff
                for i_eig in range(len(eigen_vals))
            ])
        elif lower_absolute_cutoff is not None:
            above_cutoff = eigen_vals > lower_absolute_cutoff
        elif lower_relative_cutoff is not None:
            above_cutoff = rel_eigen_vals > lower_relative_cutoff
        else:
            raise Exception('Need a lower cutoff (aboslute or relative, '
                            'e.g., 1e-15, to compute separable directions.')

        # restrict to those above cutoff
        eigen_vals = eigen_vals[above_cutoff]
        eigen_vectors = eigen_vectors[:, above_cutoff]
        # apply normlization
        if normalize:
            eigen_vals = rel_eigen_vals
        return eigen_vals, eigen_vectors

    # Identifiable directions are wanted: an filtering of the inverse
    # eigenvalues is needed (upper pass of inverse = lower pass of original)
    if inv_upper_absolute_cutoff is not None and \
            inv_upper_relative_cutoff is not None:
        below_cutoff = np.array([
            eigen_vals[i_eig] < 1 / eigen_vals > inv_upper_absolute_cutoff and
            1 / rel_eigen_vals[i_eig] > inv_upper_relative_cutoff
            for i_eig in range(len(eigen_vals))
        ])
    elif inv_upper_absolute_cutoff is not None:
        below_cutoff = 1 / eigen_vals > inv_upper_absolute_cutoff
    elif lower_relative_cutoff is not None:
        below_cutoff = 1 / rel_eigen_vals > inv_upper_relative_cutoff
    else:
        raise Exception('Need an inverse upper cutoff (aboslute or relative, '
                        'e.g., 1e-15, to compute identifiable directions.')

    # restrict to those below cutoff
    eigen_vals = eigen_vals[below_cutoff]
    eigen_vectors = eigen_vectors[:, below_cutoff]
    # apply normlization
    if normalize:
        eigen_vals = rel_eigen_vals
    return eigen_vals, eigen_vectors

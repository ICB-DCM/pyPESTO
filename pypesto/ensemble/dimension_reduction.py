import numpy as np
from typing import Callable, Union, Tuple

from .ensemble import Ensemble, EnsemblePrediction
from .utils import get_prediction_dataset

try:
    import sklearn.decomposition
    from sklearn.preprocessing import StandardScaler
    import umap
    import umap.plot
except ImportError:
    pass


def get_umap_representation_parameters(
        ens: Ensemble,
        n_components: int = 2,
        normalize_data: bool = False,
        **kwargs) -> Tuple:
    """
    Compute the representation with reduced dimensionality via umap
    (with a given number of umap components) of the parameter ensemble.
    Allows to pass on additional keyword arguments to the umap routine.

    Parameters
    ==========
    ens:
        Ensemble objects containing a set of parameter vectors

    n_components:
        number of components for the dimension reduction

    normalize_data:
        flag indicating whether the parameter ensemble should be rescaled with
        mean and standard deviation

    Returns
    =======
    umap_components:
        first components of the umap embedding

    umap_object:
        returned fitted umap object from umap.UMAP()
    """

    # call lowlevel routine using the parameter vector ensemble
    return _get_umap_representation_lowlevel(
        dataset=ens.x_vectors.transpose(),
        n_components=n_components,
        normalize_data=normalize_data,
        **kwargs
    )


def get_umap_representation_predictions(
        ens: Union[Ensemble, EnsemblePrediction],
        prediction_index: int = 0,
        n_components: int = 2,
        normalize_data: bool = False,
        **kwargs) -> Tuple:
    """
    Compute the representation with reduced dimensionality via umap
    (with a given number of umap components) of the ensemble predictions.
    Allows to pass on additional keyword arguments to the umap routine.

    Parameters
    ==========
    ens:
        Ensemble objects containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions

    prediction_index:
        index telling which prediction from the list should be analyzed

    n_components:
        number of components for the dimension reduction

    normalize_data:
        flag indicating whether the parameter ensemble should be rescaled with
        mean and standard deviation

    Returns
    =======
    umap_components:
        first components of the umap embedding

    umap_object:
        returned fitted umap object from umap.UMAP()
    """

    # extract the an array of predictions from either an Ensemble object or an
    # EnsemblePrediction object
    dataset = get_prediction_dataset(ens, prediction_index)

    # call lowlevel routine using the prediction ensemble
    return _get_umap_representation_lowlevel(
        dataset=dataset,
        n_components=n_components,
        normalize_data=normalize_data,
        **kwargs
    )


def get_pca_representation_parameters(
        ens: Ensemble,
        n_components: int = 2,
        rescale_data: bool = True,
        rescaler: Union[Callable, None] = None
) -> Tuple:
    """
    Compute the representation with reduced dimensionality via principal
    component analysis (with a given number of principal components) of the
    parameter ensemble.

    Parameters
    ==========
    ens:
        Ensemble objects containing a set of parameter vectors

    n_components:
        number of components for the dimension reduction

    rescale_data:
        flag indicating whether the principal components should be rescaled
        using a rescaler function (e.g., an arcsinh function)

    rescaler:
        callable function to rescale the output of the PCA (defaults to
        numpy.arcsinh)

    Returns
    =======
    principal_components:
        principal components of the parameter vector ensemble

    pca_object:
        returned fitted pca object from sklearn.decomposition.PCA()
    """

    return _get_pca_representation_lowlevel(
        dataset=ens.x_vectors.transpose(),
        n_components=n_components,
        rescale_data=rescale_data,
        rescaler=rescaler
    )


def get_pca_representation_predictions(
        ens: Union[Ensemble, EnsemblePrediction],
        prediction_index: int = 0,
        n_components: int = 2,
        rescale_data: bool = True,
        rescaler: Union[Callable, None] = None
) -> Tuple:
    """
    Compute the representation with reduced dimensionality via principal
    component analysis (with a given number of principal components) of the
    ensemble prediction.

    Parameters
    ==========
    ens:
        Ensemble objects containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions

    prediction_index:
        index telling which prediction from the list should be analyzed

    n_components:
        number of components for the dimension reduction

    rescale_data:
        flag indicating whether the principal components should be rescaled
        using a rescaler function (e.g., an arcsinh function)

    rescaler:
        callable function to rescale the output of the PCA (defaults to
        numpy.arcsinh)

    Returns
    =======
    principal_components:
        principal components of the parameter vector ensemble

    pca_object:
        returned fitted pca object from sklearn.decomposition.PCA()
    """

    # extract the an array of predictions from either an Ensemble object or an
    # EnsemblePrediction object
    dataset = get_prediction_dataset(ens, prediction_index)

    # call lowlevel routine using the prediction ensemble
    return _get_pca_representation_lowlevel(
        dataset=dataset,
        n_components=n_components,
        rescale_data=rescale_data,
        rescaler=rescaler
    )


def _get_umap_representation_lowlevel(
        dataset: np.ndarray,
        n_components: int = 2,
        normalize_data: bool = False,
        **kwargs) -> Tuple:
    """
    Compute the representation with reduced dimensionality via principal
    component analysis (with a given number of principal components) of the
    parameter ensemble.

    Parameters
    ==========
    dataset:
        numpy array containing either the ensemble predictions or the parameter
        ensemble itself

    n_components:
        number of components for the dimension reduction

    rescale_data:
        flag indicating whether the principal components should be rescaled
        using a rescaler function (e.g., an arcsinh function)

    rescaler:
        callable function to rescale the output of the PCA (defaults to
        numpy.arcsinh)

    Returns
    =======
    umap_components:
        first components of the umap embedding

    umap_object:
        returned fitted umap object from umap.UMAP()
    """

    # create a umap object
    umap_object = umap.UMAP(n_components=n_components, **kwargs)

    # normalize data with mean and standard deviation if wanted
    if normalize_data:
        dataset = StandardScaler().fit_transform(dataset)

    # perform the manifold fitting and transform the dataset
    umap_components = umap_object.fit_transform(dataset)

    return umap_components, umap_object


def _get_pca_representation_lowlevel(
        dataset: np.ndarray,
        n_components: int = 2,
        rescale_data: bool = True,
        rescaler: Union[Callable, None] = None
) -> Tuple:
    """
    Compute the representation with reduced dimensionality via principal
    component analysis (with a given number of principal components) of the
    parameter ensemble.

    Parameters
    ==========
    dataset:
        numpy array containing either the ensemble predictions or the parameter
        ensemble itself

    n_components:
        number of components for the dimension reduction

    rescale_data:
        flag indicating whether the principal components should be rescaled
        using a rescaler function (e.g., an arcsinh function)

    rescaler:
        callable function to rescale the output of the PCA (defaults to
        numpy.arcsinh)

    Returns
    =======
    principal_components:
        principal components of the parameter vector ensemble

    pca_object:
        returned fitted pca object from sklearn.decomposition.PCA()
    """

    # create a PCA object and decompose the dataset
    pca_object = sklearn.decomposition.PCA(n_components=n_components)
    pca_object.fit(dataset)
    # get the projection down to the first components
    principal_components = pca_object.transform(dataset)

    # rescale the principal components with a non-linear function, if wanted
    if rescale_data:
        if rescaler is None:
            # use arcsinh as default
            principal_components = np.arcsinh(principal_components)
        else:
            # use provided funcation for rescaling
            principal_components = rescaler(principal_components)

    return principal_components, pca_object

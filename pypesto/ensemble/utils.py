import h5py
import numpy as np
import pandas as pd
import os
from typing import Callable, Union
from pathlib import Path

from .constants import (EnsembleType, OUTPUT, UPPER_BOUND, LOWER_BOUND,
                        PREDICTION_RESULTS, PREDICTION_ID, SUMMARY,
                        OPTIMIZE, SAMPLE)
from .ensemble import (Ensemble, EnsemblePrediction)
from ..store import read_result


def read_from_csv(path: str,
                  sep: str = '\t',
                  index_col: int = 0,
                  headline_parser: Callable = None,
                  ensemble_type: EnsembleType = None,
                  lower_bound: np.ndarray = None,
                  upper_bound: np.ndarray = None):
    """
    function for creating an ensemble from a csv file

    Parameters
    ----------
    path:
        path to csv file to read in parameter ensemble
    sep:
        separator in csv file
    index_col:
        index column in csv file
    headline_parser:
        A function which reads in the headline of the csv file and converts it
        into vector_tags (see constructor of Ensemble for more details)
    ensemble_type:
        Ensemble type: representative sample or random ensemble
    lower_bound:
        array of potential lower bounds for the parameters
    upper_bound:
        array of potential upper bounds for the parameters

    Returns
    -------
    result:
        Ensemble object of parameter vectors
    """
    # get the data from the csv
    ensemble_df = pd.read_csv(path, sep=sep, index_col=index_col)

    # set the type of the ensemble
    if ensemble_type is None:
        ensemble_type = EnsembleType.ensemble

    return read_from_df(dataframe=ensemble_df,
                        headline_parser=headline_parser,
                        ensemble_type=ensemble_type,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound)


def read_ensemble_from_hdf5(filename: str,
                            input_type: str = OPTIMIZE,
                            remove_burn_in: bool = True,
                            chain_slice: slice = None,
                            cutoff: float = np.inf,
                            max_size: int = np.inf
                            ):
    """
    Create an ensemble from an HDF5 storage file.

    Parameters
    ----------
    filename:
        Name or path of the HDF5 file.
    input_type:
        Which type of ensemble to create. From History, from
        Optimization or from Sample.

    Returns:
    -------
    ensemble:
        Ensemble object of parameter vectors
    """
    # TODO: add option HISTORY. Need to fix
    #  reading history from hdf5.
    if input_type == OPTIMIZE:
        result = read_result(filename=filename,
                             optimize=True)
        return Ensemble.from_optimization_endpoints(result=result,
                                                    cutoff=cutoff,
                                                    max_size=max_size)
    elif input_type == SAMPLE:
        result = read_result(filename=filename,
                             sample=True)
        return Ensemble.from_sample(result=result,
                                    remove_burn_in=remove_burn_in,
                                    chain_slice=chain_slice)
    else:
        raise ValueError('The type you provided was neither '
                         f'"{SAMPLE}" nor "{OPTIMIZE}". Those are '
                         'currently the only supported types. '
                         'Please choose one of them.')


def read_from_df(dataframe: pd.DataFrame,
                 headline_parser: Callable = None,
                 ensemble_type: EnsembleType = None,
                 lower_bound: np.ndarray = None,
                 upper_bound: np.ndarray = None):
    """
    function for creating an ensemble from a csv file

    Parameters
    ----------
    dataframe:
        pandas.DataFrame to read in parameter ensemble
    headline_parser:
        A function which reads in the headline of the csv file and converts it
        into vector_tags (see constructor of Ensemble for more details)
    ensemble_type:
        Ensemble type: representative sample or random ensemble
    lower_bound:
        array of potential lower bounds for the parameters
    upper_bound:
        array of potential upper bounds for the parameters

    Returns
    -------
    result:
        Ensemble object of parameter vectors
    """
    # if we have a parser to make vector_tags from column names, we use it
    vector_tags = None
    if headline_parser is not None:
        vector_tags = headline_parser(list(dataframe.columns))
    # set the type of the ensemble
    if ensemble_type is None:
        ensemble_type = EnsembleType.ensemble

    return Ensemble(x_vectors=dataframe.values,
                    x_names=list(dataframe.index),
                    vector_tags=vector_tags,
                    ensemble_type=ensemble_type,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound)


def write_ensemble_prediction_to_h5(ensemble_prediction: EnsemblePrediction,
                                    output_file: str,
                                    base_path: str = None):
    # parse base path
    base = Path('')
    if base_path is not None:
        base = Path(base_path)

    # open file
    f = h5py.File(output_file, 'w')

    # write prediction ID if available
    if ensemble_prediction.prediction_id is not None:
        f.create_dataset(os.path.join(base, PREDICTION_ID),
                         data=ensemble_prediction.prediction_id)

    # write the single prediction results
    for i_result, result in enumerate(ensemble_prediction.prediction_results):
        tmp_base_path = os.path.join(base, f'{PREDICTION_RESULTS}_{i_result}')
        result.write_to_h5(output_file, base_path=tmp_base_path)

    # write lower bounds per condition, if available
    if ensemble_prediction.lower_bound is not None:
        f.create_group(os.path.join(base, f'{LOWER_BOUND}s'))
        for i_cond, lower_bounds in enumerate(ensemble_prediction.lower_bound):
            condition_id = \
                ensemble_prediction.prediction_results[0].condition_ids[i_cond]
            f.create_group(os.path.join(base, condition_id))
            f.create_dataset(os.path.join(base, condition_id, LOWER_BOUND),
                             data=lower_bounds)

    # write upper bounds per condition, if available
    if ensemble_prediction.upper_bound is not None:
        f.create_group(os.path.join(base, f'{UPPER_BOUND}s'))
        for i_cond, upper_bounds in enumerate(ensemble_prediction.upper_bound):
            condition_id = \
                ensemble_prediction.prediction_results[0].condition_ids[i_cond]
            f.create_group(os.path.join(base, condition_id))
            f.create_dataset(os.path.join(base, condition_id, UPPER_BOUND),
                             data=upper_bounds)

    # write summary statistics to h5 file
    for i_key in ensemble_prediction.prediction_summary.keys():
        i_summary = ensemble_prediction.prediction_summary[i_key]
        if i_summary is not None:
            tmp_base_path = os.path.join(base, f'{SUMMARY}_{i_key}')
            f.create_group(tmp_base_path)
            i_summary.write_to_h5(output_file, base_path=tmp_base_path)

    # close file
    f.close()


def get_prediction_dataset(ens: Union[Ensemble, EnsemblePrediction],
                           prediction_index: int = 0) -> np.ndarray:
    """
    Extract an array of prediction from either an Ensemble object which
    contains a list of predictions of from an EnsemblePrediction object.

    Parameters
    ==========
    ens:
        Ensemble objects containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions

    prediction_index:
        index telling which prediction from the list should be analyzed

    Returns
    =======
    dataset:
        numpy array containing the ensemble predictions
    """

    if isinstance(ens, Ensemble):
        dataset = ens.predictions[prediction_index]
    elif isinstance(ens, EnsemblePrediction):
        ens.condense_to_arrays()
        dataset = ens.prediction_arrays[OUTPUT].transpose()
    else:
        raise Exception('Need either an Ensemble object with predictions or '
                        'an EnsemblePrediction object as input. Stopping.')

    return dataset

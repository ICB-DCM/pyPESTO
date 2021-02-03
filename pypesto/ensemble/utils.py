import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable, Dict

from ..prediction import PredictionResult, PredictionConditionResult
from .constants import (PREDICTOR, PREDICTION_ID, PREDICTION_RESULTS,
                        PREDICTION_ARRAYS, PREDICTION_SUMMARY, OUTPUT,
                        OUTPUT_SENSI, TIMEPOINTS, X_VECTOR, NX, X_NAMES,
                        NVECTORS, VECTOR_TAGS, PREDICTIONS, MODE_FUN,
                        EnsembleType, ENSEMBLE_TYPE, MEAN, MEDIAN,
                        STANDARD_DEVIATION, PERCENTILE, SUMMARY, LOWER_BOUND,
                        UPPER_BOUND)
from .ensemble import (Ensemble, EnsemblePrediction)



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
    # if we have a parser to make vector_tags from column names, we use it
    vector_tags = None
    if headline_parser is not None:
        vector_tags = headline_parser(list(ensemble_df.columns))
    # set the type of the ensemble
    if ensemble_type is None:
        ensemble_type = EnsembleType.ensemble

    return read_from_df(dataframe=ensemble_df,
                        headline_parser=headline_parser,
                        ensemble_type=ensemble_type,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound)


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
                                    base_path: str= None):
    f = h5py.File(output_file, 'w')

    if ensemble_prediction.prediction_id is not None:
        f.create_group('prediction_id', data=ensemble_prediction.prediction_id)
    for i_result, result in enumerate(ensemble_prediction.prediction_results):
        base_path = f'PredictionResult_{i_result}'
        result.write_to_h5(output_file, base_path=base_path)
    f.close()

def write_ensemble_to_h5():
    pass

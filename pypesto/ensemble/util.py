"""Ensemble utilities."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal, Union

import h5py
import numpy as np
import pandas as pd

from ..C import (
    LOWER_BOUND,
    OPTIMIZE,
    OUTPUT,
    OUTPUT_IDS,
    OUTPUT_SIGMAY,
    OUTPUT_WEIGHT,
    PREDICTION_ID,
    PREDICTION_RESULTS,
    SAMPLE,
    SUMMARY,
    TIMEPOINTS,
    UPPER_BOUND,
    X_NAMES,
    EnsembleType,
)
from ..result import PredictionConditionResult, PredictionResult
from ..store import read_result, write_array
from .ensemble import Ensemble, EnsemblePrediction


def read_from_csv(
    path: str,
    sep: str = "\t",
    index_col: int = 0,
    headline_parser: Callable = None,
    ensemble_type: EnsembleType = None,
    lower_bound: np.ndarray = None,
    upper_bound: np.ndarray = None,
):
    """
    Create an ensemble from a csv file.

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

    return read_from_df(
        dataframe=ensemble_df,
        headline_parser=headline_parser,
        ensemble_type=ensemble_type,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def read_ensemble_from_hdf5(
    filename: str,
    input_type: Literal["optimize", "sample"] = OPTIMIZE,
    remove_burn_in: bool = True,
    chain_slice: slice = None,
    cutoff: float = np.inf,
    max_size: int = np.inf,
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

    Returns
    -------
    ensemble:
        Ensemble object of parameter vectors
    """
    # TODO: add option HISTORY. Need to fix
    #  reading history from hdf5.
    if input_type == OPTIMIZE:
        result = read_result(filename=filename, optimize=True)
        return Ensemble.from_optimization_endpoints(
            result=result, rel_cutoff=cutoff, max_size=max_size
        )
    elif input_type == SAMPLE:
        result = read_result(filename=filename, sample=True)
        return Ensemble.from_sample(
            result=result,
            remove_burn_in=remove_burn_in,
            chain_slice=chain_slice,
        )
    else:
        raise ValueError(
            "The type you provided was neither "
            f'"{SAMPLE}" nor "{OPTIMIZE}". Those are '
            "currently the only supported types. "
            "Please choose one of them."
        )


def read_from_df(
    dataframe: pd.DataFrame,
    headline_parser: Callable = None,
    ensemble_type: EnsembleType = None,
    lower_bound: np.ndarray = None,
    upper_bound: np.ndarray = None,
):
    """
    Create an ensemble from a csv file.

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

    return Ensemble(
        x_vectors=dataframe.values,
        x_names=list(dataframe.index),
        vector_tags=vector_tags,
        ensemble_type=ensemble_type,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def write_ensemble_prediction_to_h5(
    ensemble_prediction: EnsemblePrediction,
    output_file: str,
    base_path: str = None,
):
    """
    Write an `EnsemblePrediction` to hdf5.

    Parameters
    ----------
    ensemble_prediction:
        The prediciton to be saved.
    output_file:
        The filename of the hdf5 file.
    base_path:
        An optional filepath where the file should be saved to.
    """
    # parse base path
    base = Path("")
    if base_path is not None:
        base = Path(base_path)

    # open file
    with h5py.File(output_file, "a") as f:
        # write prediction ID if available
        if ensemble_prediction.prediction_id is not None:
            f.create_dataset(
                os.path.join(base, PREDICTION_ID),
                data=ensemble_prediction.prediction_id,
            )

        # write lower bounds per condition, if available
        if ensemble_prediction.lower_bound is not None:
            if isinstance(ensemble_prediction.lower_bound[0], np.ndarray):
                lb_grp = f.require_group(LOWER_BOUND)
                for i_cond, lower_bounds in enumerate(
                    ensemble_prediction.lower_bound
                ):
                    condition_id = ensemble_prediction.prediction_results[
                        0
                    ].condition_ids[i_cond]
                    write_array(lb_grp, condition_id, lower_bounds)
            elif isinstance(ensemble_prediction.lower_bound[0], float):
                f.create_dataset(
                    LOWER_BOUND, data=ensemble_prediction.lower_bound
                )

        # write upper bounds per condition, if available
        if ensemble_prediction.upper_bound is not None:
            if isinstance(ensemble_prediction.upper_bound[0], np.ndarray):
                ub_grp = f.require_group(UPPER_BOUND)
                for i_cond, upper_bounds in enumerate(
                    ensemble_prediction.upper_bound
                ):
                    condition_id = ensemble_prediction.prediction_results[
                        0
                    ].condition_ids[i_cond]
                    write_array(ub_grp, condition_id, upper_bounds)
            elif isinstance(ensemble_prediction.upper_bound[0], float):
                f.create_dataset(
                    UPPER_BOUND, data=ensemble_prediction.upper_bound
                )

        # write summary statistics to h5 file
        for (
            summary_id,
            summary,
        ) in ensemble_prediction.prediction_summary.items():
            if summary is None:
                continue
            tmp_base_path = os.path.join(base, f"{SUMMARY}_{summary_id}")
            f.create_group(tmp_base_path)
            summary.write_to_h5(output_file, base_path=tmp_base_path)

        # write the single prediction results
        for i_result, result in enumerate(
            ensemble_prediction.prediction_results
        ):
            tmp_base_path = os.path.join(
                base, f"{PREDICTION_RESULTS}_{i_result}"
            )
            result.write_to_h5(output_file, base_path=tmp_base_path)


def get_prediction_dataset(
    ens: Union[Ensemble, EnsemblePrediction], prediction_index: int = 0
) -> np.ndarray:
    """
    Extract an array of prediction.

    Can be done from either an Ensemble object which contains a list of
    predictions of from an EnsemblePrediction object.

    Parameters
    ----------
    ens:
        Ensemble objects containing a set of parameter vectors and a set of
        predictions or EnsemblePrediction object containing only predictions

    prediction_index:
        index telling which prediction from the list should be analyzed

    Returns
    -------
    dataset:
        numpy array containing the ensemble predictions
    """
    if isinstance(ens, Ensemble):
        dataset = ens.predictions[prediction_index]
    elif isinstance(ens, EnsemblePrediction):
        ens.condense_to_arrays()
        dataset = ens.prediction_arrays[OUTPUT].transpose()
    else:
        raise Exception(
            "Need either an Ensemble object with predictions or "
            "an EnsemblePrediction object as input. Stopping."
        )

    return dataset


def read_ensemble_prediction_from_h5(
    predictor: Union[Callable[[Sequence], PredictionResult], None],
    input_file: str,
):
    """Read an ensemble prediction from an HDF5 File."""
    # open file
    with h5py.File(input_file, "r") as f:
        pred_res_list = []
        bounds = {}
        for key in f.keys():
            if key.startswith(SUMMARY):
                continue
            if key == PREDICTION_ID:
                prediction_id = f[key][()].decode()
                continue
            if key in {LOWER_BOUND, UPPER_BOUND}:
                if isinstance(f[key], h5py._hl.dataset.Dataset):
                    bounds[key] = f[key][:]
                    continue
                bounds[key] = [
                    f[f"{key}/{cond}"][()] for cond in f[key].keys()
                ]
                bounds[key] = np.array(bounds[key])
                continue
            x_names = list(decode_array(f[f"{key}/{X_NAMES}"][()]))
            condition_ids = list(decode_array(f[f"{key}/condition_ids"][()]))
            pred_cond_res_list = []
            for id, _ in enumerate(condition_ids):
                output = f[f"{key}/{id}/{OUTPUT}"][:]
                output_ids = tuple(
                    decode_array(f[f"{key}/{id}/{OUTPUT_IDS}"][:])
                )
                timepoints = f[f"{key}/{id}/{TIMEPOINTS}"][:]
                try:
                    output_weight = f[f"{key}/{id}/{OUTPUT_WEIGHT}"][()]
                except KeyError:
                    output_weight = None
                try:
                    output_sigmay = f[f"{key}/{id}/{OUTPUT_SIGMAY}"][:]
                except KeyError:
                    output_sigmay = None
                pred_cond_res_list.append(
                    PredictionConditionResult(
                        timepoints=timepoints,
                        output_ids=output_ids,
                        output=output,
                        x_names=x_names,
                        output_weight=output_weight,
                        output_sigmay=output_sigmay,
                    )
                )
            pred_res_list.append(
                PredictionResult(
                    conditions=pred_cond_res_list, condition_ids=condition_ids
                )
            )
        return EnsemblePrediction(
            predictor=predictor,
            prediction_id=prediction_id,
            prediction_results=pred_res_list,
        )


def decode_array(array: np.ndarray) -> np.ndarray:
    """Decode array of bytes to string."""
    for i in range(len(array)):
        array[i] = array[i].decode()
    return array

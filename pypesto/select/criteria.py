import math


# TODO should fixed parameters count as measurements when comparing to models
#      that estimate the same parameters?
def calculate_aic(n_estimated: int, nllh: float) -> float:
    """
    Calculate the Akaike information criterion for a model.

    Arguments
    ---------
    n_estimated:
        The number of estimated parameters in the model.
    nllh:
        The negative log likelihood,
        e.g.: the `optimize_result.list[0]['fval']` attribute of the object
        returned by `pypesto.minimize`.
    """
    return 2*(n_estimated + nllh)


def calculate_aicc(
        n_estimated: int,
        nllh: float,
        n_measurements: int,
        n_priors: int,
) -> float:
    """
    Calculate the Akaike information criterion for a model.

    Arguments
    ---------
    n_estimated:
        The number of estimated parameters in the model.
    nllh:
        The negative log likelihood,
        e.g.: the `optimize_result.list[0]['fval']` attribute of the object
        returned by `pypesto.minimize`.
    n_measurements:
        The number of measurements used in the objective function of the model,
        e.g.: `len(petab_problem.measurement_df)`.
    n_priors:
        The number of priors used in the objective function of the model,
        e.g.: `len(pypesto_problem.x_priors._objectives)`.
        TODO make public property for number of priors in objective? or in
             problem, since `x_priors` == None is possible.
    """
    # TODO untested
    return (
        calculate_aic(n_estimated, nllh)
        + 2*n_estimated*(n_estimated + 1)
        / (n_measurements + n_priors - n_estimated - 1)
    )


def calculate_bic(
        n_estimated: int,
        nllh: float,
        n_measurements: int,
        n_priors: int,
):
    """
    Calculate the Bayesian information criterion for a model.

    Arguments
    ---------
    n_estimated:
        The number of estimated parameters in the model.
    nllh:
        The negative log likelihood,
        e.g.: the `optimize_result.list[0]['fval']` attribute of the object
        returned by `pypesto.minimize`.
    n_measurements:
        The number of measurements used in the objective function of the model,
        e.g.: `len(petab_problem.measurement_df)`.
    n_priors:
        The number of priors used in the objective function of the model,
        e.g.: `len(pypesto_problem.x_priors._objectives)`.
        TODO make public property for number of priors in objective? or in
             problem, since `x_priors` == None is possible.
    """
    # TODO untested
    return n_estimated*math.log(n_measurements + n_priors) + 2*nllh

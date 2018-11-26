import os
import numpy as np
import copy
import logging
import pandas as pd
from .objective import Objective
from .constants import MODE_FUN, MODE_RES, HESS

try:
    import amici
except ImportError:
    amici = None

logger = logging.getLogger(__name__)


class AmiciObjective(Objective):
    """
    This is a convenience class to compute an objective function from an
    AMICI model.

    Parameters
    ----------

    amici_model: amici.Model
        The amici model.

    amici_solver: amici.Solver
        The solver to use for the numeric integration of the model.

    edata:
        The experimental data.

    max_sensi_order: int
        Maximum sensitivity order supported by the model.
    """

    def __init__(self, amici_model, amici_solver, edata, max_sensi_order=None,
                 preprocess_edata=True, options=None):
        if amici is None:
            raise ImportError('This objective requires an installation of '
                              'amici (github.com/icb-dcm/amici. Install via '
                              'pip3 install amici.')

        if max_sensi_order is None:
            max_sensi_order = 2 if amici_model.o2mode else 1

        fun = self.get_bound_fun()

        if max_sensi_order > 0:
            grad = True
            hess = True
        else:
            grad = None
            hess = None

        res = self.get_bound_res()

        if max_sensi_order > 0:
            sres = True
        else:
            sres = None

        super().__init__(
            fun=fun, grad=grad, hess=hess, hessp=None,
            res=res, sres=sres,
            fun_accept_sensi_orders=True,
            res_accept_sensi_orders=True,
            options=options
        )

        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.dim = len(amici_model.getParameterList())

        if preprocess_edata:
            self.preequilibration_edata = dict()
            self.preprocess_edata(edata)
            self.edata = edata
        else:
            self.edata = edata
            self.preequilibration_edata = None

        self.max_sensi_order = max_sensi_order

        # extract parameter names from model
        self.x_names = list(self.amici_model.getParameterNames())

    def get_bound_fun(self):
        """
        Generate a fun function that calls _call_amici with MODE_FUN. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def fun(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_FUN)

        return fun

    def get_bound_res(self):
        """
        Generate a res function that calls _call_amici with MODE_RES. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def res(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_RES)

        return res

    def rebind_fun(self):
        """
        Replace the current fun function with one that is bound to the current
        instance
        """
        self.fun = self.get_bound_fun()

    def rebind_res(self):
        """
        Replace the current res function with one that is bound to the current
        instance
        """
        self.res = self.get_bound_res()

    def __deepcopy__(self, memodict=None):
        model = amici.ModelPtr(self.amici_model.clone())
        solver = amici.SolverPtr(self.amici_solver.clone())
        edata = [amici.ExpData(data) for data in self.edata]
        other = AmiciObjective(model, solver, edata)
        for attr in self.__dict__:
            if attr not in ['amici_solver', 'amici_model', 'edata', 'preequilibration_edata']:
                other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    def update_from_problem(self,
                            dim_full,
                            x_free_indices,
                            x_fixed_indices,
                            x_fixed_vals):
        """
        Handle fixed parameters. Here we implement the amici exclusive
        initialization of ParameterLists and respectively replace the
        generic postprocess function

        Parameters
        ----------

        dim_full: int
            Dimension of the full vector including fixed parameters.

        x_free_indices: array_like of int
            Vector containing the indices (zero-based) of free parameters
            (complimentary to x_fixed_indices).

        x_fixed_indices: array_like of int, optional
            Vector containing the indices (zero-based) of parameter components
            that are not to be optimized.

        x_fixed_vals: array_like, optional
            Vector of the same length as x_fixed_indices, containing the values
            of the fixed parameters.
        """
        super(AmiciObjective, self).update_from_problem(
            dim_full,
            x_free_indices,
            x_fixed_indices,
            x_fixed_vals
        )
        # we subindex the existing plist in case there already is a user
        # specified plist
        plist = self.amici_model.getParameterList()
        plist = [plist[idx] for idx in x_free_indices]
        self.amici_model.setParameterList(plist)

        self.dim = len(plist)

        def postprocess(result):
            if HESS in result:
                hess = result[HESS]
                if hess.shape[0] == dim_full:
                    # see https://github.com/ICB-DCM/AMICI/issues/274
                    hess = hess[..., x_free_indices]
                    result[HESS] = hess
            return result

        self.postprocess = postprocess

        # now we need to rebind fun and res to this instance of AmiciObjective
        # for the changes to have an effect
        self.rebind_fun()
        self.rebind_res()

    def _call_amici(
            self,
            x,
            sensi_orders,
            mode
    ):
        # amici is built so that only the maximum sensitivity is required,
        # the lower orders are then automatically computed

        # gradients can always be computed
        sensi_order = min(max(sensi_orders), 1)
        # order 2 currently not implemented, we are using the FIM

        # check if sensitivities can be computed
        if sensi_order > self.max_sensi_order:
            raise Exception("Sensitivity order not allowed.")

        # TODO: For large-scale models it might be bad to always reserve
        # space in particular for the Hessian.
        nllh = 0.0
        snllh = np.zeros(self.dim)
        ssnllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        # set parameters in model
        self.amici_model.setParameters(x)

        # set order in solver
        self.amici_solver.setSensitivityOrder(sensi_order)

        if self.preequilibration_edata:
            preeq_status = self.run_preequilibration(sensi_orders, mode)
            if preeq_status is not None:
                return preeq_status

        # loop over experimental data
        for data_index, data in enumerate(self.edata):

            if self.preequilibration_edata:
                original_value_dict = self.preprocess_preequilibration(data)
            else:
                original_value_dict = None

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                data)

            if self.preequilibration_edata:
                self.postprocess_preequilibration(data, original_value_dict)

            # logging
            logger.debug(f'=== DATASET {data_index} ===')
            logger.debug(f'status: {rdata["status"]}')
            logger.debug(f'llh: {rdata["llh"]}')

            if 't_steadystate' in rdata and rdata['t_steadystate'] != np.nan:
                logger.debug(f't_steadystate: {rdata["t_steadystate"]}')
            logger.debug(f'res: {rdata["res"]}')

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            # extract required result fields
            if mode == MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    snllh -= rdata['sllh']
                    # TODO: Compute the full Hessian, and check here
                    ssnllh -= rdata['FIM']

            elif mode == MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    sres = np.vstack([sres, rdata['sres']]) \
                        if sres.size else rdata['sres']

        # map_to_output is called twice, might be prettified
        return Objective.output_to_tuple(
            sensi_orders,
            mode,
            fval=nllh, grad=snllh, hess=ssnllh,
            res=res, sres=sres
        )

    def preprocess_preequilibration(self, data):
        original_fixed_parameters_preequilibration = None
        original_initial_states = None
        original_initial_state_sensitivities = None
        if len(data.fixedParametersPreequilibration):
            original_initial_states = self.amici_model.getInitialStates()

            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                original_initial_state_sensitivities = \
                    self.amici_model.getInitialStateSensitivities()

            fixed_parameters = copy.deepcopy(
                list(data.fixedParametersPreequilibration)
            )
            data.fixedParametersPreequilibration = []
            original_fixed_parameters_preequilibration = fixed_parameters

            self.amici_model.setInitialStates(
                self.preequilibration_edata[str(fixed_parameters)]['x0']
            )
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                self.amici_model.setInitialStateSensitivities(
                    self.preequilibration_edata[
                        str(fixed_parameters)
                    ]['sx0'].flatten()
                )

        return {
            'k': original_fixed_parameters_preequilibration,
            'x0': original_initial_states,
            'sx0': original_initial_state_sensitivities
        }

    def postprocess_preequilibration(self, data, original_value_dict):
        if original_value_dict['k']:
            data.fixedParametersPreequilibration = original_value_dict['k']

        if original_value_dict['x0']:
            self.amici_model.setInitialStates(original_value_dict['x0'])

        if original_value_dict['sx0']:
            self.amici_model.setInitialStateSensitivities(
                original_value_dict['sx0']
            )

    def preprocess_edata(self, edata_vector):
        for edata in edata_vector:
            fixed_parameters = list(edata.fixedParametersPreequilibration)
            if str(fixed_parameters) in self.preequilibration_edata.keys() or \
               len(fixed_parameters) == 0:
                continue  # we only need to keep unique ones

            preeq_edata = amici.amici.ExpData(self.amici_model.get())
            preeq_edata.fixedParametersPreequilibration = fixed_parameters

            # only preequilibration
            preeq_edata.setTimepoints([])

            self.preequilibration_edata[str(fixed_parameters)] = dict(
                edata=preeq_edata
            )

    def get_error_output(self, sensi_orders, mode):
        if not self.amici_model.nt():
            nt = sum([data.nt() for data in self.edata])
        else:
            nt = sum([data.nt() if data.nt() else self.amici_model.nt()
                      for data in self.edata])
        n_res = nt * self.amici_model.nytrue
        return Objective.output_to_tuple(
            sensi_orders=sensi_orders,
            mode=mode,
            fval=np.inf,
            grad=np.nan * np.ones(self.dim),
            hess=np.nan * np.ones([self.dim, self.dim]),
            res=np.nan * np.ones(n_res),
            sres=np.nan * np.ones([n_res, self.dim])
        )

    def run_preequilibration(self, sensi_orders, mode):
        """
        Run preequilibration.
        """

        for fixedParameters in self.preequilibration_edata:
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                self.preequilibration_edata[fixedParameters]['edata'])

            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            self.preequilibration_edata[fixedParameters]['x0'] = \
                rdata['x0']
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                self.preequilibration_edata[fixedParameters]['sx0'] = \
                    rdata['sx0']


    def simulations_to_measurement_df(self, x, measurement_file=None):
        """
        Simulate all conditions.
        """

        self.amici_model.setParameters(amici.DoubleVector(x))
        self.amici_solver.setSensitivityOrder(0)

        if self.preequilibration_edata:
            preeq_status = self.run_preequilibration(0, MODE_FUN)
            if preeq_status is not None:
                raise ValueError(preeq_status)

        #df_mes = pd.read_csv(measurement_file, sep='\t')
        df_sim = pd.DataFrame(columns=['observableId',
                                   'preequilibrationConditionId',
                                   'simulationConditionId',
                                   'measurement',
                                   'time',
                                   'observableParameters',
                                   'noiseParameters'
                                   'observableTransformation'])

        observable_ids = self.amici_model.getObservableIds()

        # loop over experimental data
        for data_index, data in enumerate(self.edata):

            condition_id = f'condition_{data_index}'

            if self.preequilibration_edata:
                original_value_dict = self.preprocess_preequilibration(data)
            else:
                original_value_dict = None

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                data)

            if self.preequilibration_edata:
                self.postprocess_preequilibration(data, original_value_dict)

            if rdata['status'] < amici.AMICI_SUCCESS:
                logger.debug('Simulation for condition {data_index} failed with status {rdata["status"]}')
                continue

            y = rdata['y']
            timepoints = rdata['t']

            for observable_idx in range(y.shape[1]):
                for timepoint_idx in range(y.shape[0]):
                    simulation = y[timepoint_idx, observable_idx]
                    if np.isnan(simulation):
                        continue
                    df_sim = df_sim.append(
                        {'observableId': observable_ids[observable_idx],
                         'preequilibrationConditionId': None,
                         'simulationConditionId': condition_id, # TODO Need from input file
                         'measurement': simulation,
                         'time': timepoints[timepoint_idx],
                         'observableParameters': None, # TODO Need from input file
                         'noiseParameters': rdata['sigmay'][timepoint_idx, observable_idx],
                         'observableTransformation': None}, # TODO Need from input file
                        ignore_index=True)
        return df_sim


def amici_objective_from_measurement_file(condition_filename, measurement_filename, amici_model, **kwargs):
    """
    Create AmiciObjective based on measurement and condition files.

    TODO: Does not support any condition-specific parameters yet
    """

    condition_df = pd.read_csv(condition_filename, sep='\t')
    measurement_df = pd.read_csv(measurement_filename, sep='\t')
    measurement_df.time = measurement_df.time.astype(float)

    if not np.all(measurement_df.observableParameters.isnull()):
        raise ValueError('observableParameter column currently not supported.')

    # Number of AMICI simulations will be number of unique preequilibrationCondition simulationCondition pairs
    # Can be improved by checking for identical condition vectors
    # (cannot group by NaNs)
    simulation_conditions = measurement_df.groupby(
        [measurement_df['preequilibrationConditionId'].fillna('#####'),
         measurement_df['simulationConditionId'].fillna('#####')])\
        .size().reset_index()\
        .replace({'preequilibrationConditionId':{'#####': np.nan}})\
        .replace({'simulationConditionId':{'#####': np.nan}})

    observable_ids = amici_model.getObservableIds()
    fixed_parameter_ids = amici_model.getFixedParameterIds()

    edatas = []
    for edata_idx, simulation_condition in simulation_conditions.iterrows():
        # amici.ExpData for each simulation
        cur_measurement_df = measurement_df.loc[
                             (measurement_df.preequilibrationConditionId == simulation_condition.preequilibrationConditionId)
                            & (measurement_df.simulationConditionId == simulation_condition.simulationConditionId), :]

        timepoints = sorted(cur_measurement_df.time.unique())

        edata = amici.ExpData(amici_model.get())
        edata.setTimepoints(timepoints)

        if len(fixed_parameter_ids):
            fixed_parameter_values = condition_df.loc[
                condition_df.conditionId == simulation_condition.simulationConditionId, fixed_parameter_ids].values
            edata.fixedParameters = fixed_parameter_values.astype(float).flatten()

            if simulation_condition.preequilibrationConditionId:
                fixed_preequilibration_parameter_values = condition_df.loc[
                    condition_df.conditionId == simulation_condition.preequilibrationConditionId, fixed_parameter_ids].values
                edata.fixedParametersPreequilibration = fixed_preequilibration_parameter_values.astype(float).flatten()

        y = np.full(shape=(edata.nt(), edata.nytrue()), fill_value=np.nan)
        sigma_y = np.full(shape=(edata.nt(), edata.nytrue()), fill_value=np.nan)

        # add measurements and stddev
        for i, measurement in cur_measurement_df.iterrows():
            time_idx = timepoints.index(measurement.time)
            observable_idx = observable_ids.index(f'observable_{measurement.observableId}') # TODO measurement files should contain prefix

            y[time_idx, observable_idx] = measurement.measurement

            if isinstance(measurement.noiseParameters, float):
                sigma_y[time_idx, observable_idx] = measurement.noiseParameters

        edata.setObservedData(y.flatten())
        edata.setObservedDataStdDev(sigma_y.flatten())

    edatas.append(edata)

    obj = AmiciObjective(amici_model=amici_model, edata=edatas, **kwargs)

    return obj


def import_sbml_model(sbml_model_file, model_output_dir, model_name=None,
                      measurement_file=None, condition_file=None, **kwargs):
    """
    Import SBML model following standard format documented in xxx

    Determine fixed parameters and observables from SBML file.
    """

    if not model_name:
        model_name = os.path.splitext(os.path.split(sbml_model_file)[-1])[0]

    sbml_importer = amici.SbmlImporter(sbml_model_file)

    # Determine constant parameters # TODO: also those marked constant in sbml model?
    constant_parameters = None
    if condition_file:
        condition_df = pd.read_csv(condition_file, sep='\t')
        constant_parameters = list(set(condition_df.columns.values.tolist()) - set(['conditionId', 'conditionName']))

    # Retrieve model output names and formulae from AssignmentRules and remove the respective rules
    observables = amici.assignmentRules2observables(
            sbml_importer.sbml, # the libsbml model object
            filter_function=lambda variable: variable.getId().startswith('observable_') \
                                             and not variable.getId().endswith('_sigma')
        )

    # Determine noise parameters
    sigmas = None
    if measurement_file:
        measurement_df = pd.read_csv(measurement_file, sep='\t')
        # Read sigma<->observable mapping from measurement file
        sigmas = {}
        # for easier grouping
        measurement_df.loc[measurement_df.noiseParameters.apply(isinstance, args=(float,)), 'noiseParameters'] = np.nan
        obs_noise_df = measurement_df.groupby(['observableId', 'noiseParameters']).size().reset_index()
        if len(obs_noise_df.observableId) != len(obs_noise_df.observableId.unique()):
            raise AssertionError('Different noise parameters for same output currently not supported.')

        for _, row in obs_noise_df.iterrows():
            if isinstance(row.noiseParameters, float):
                continue

            assignment_rule = sbml_importer.sbml.getAssignmentRuleByVariable(
                    f'sigma_{row.observableId}'
                ).getFormula()
            if not f'observable_{row.observableId}' in sigmas:
                sigmas[f'observable_{row.observableId}'] = assignment_rule
            elif sigmas[f'observable_{row.observableId}'] != assignment_rule:
                raise ValueError('Inconsistent sigma specified for observable_{row.observableId}: '
                                 'Previously "{sigmas[f"observable_{row.observableId}"]}", now "{assignment_rule}".')

        # Check if all observables in measurement files have been specified in the model
        measurement_observables = [f'observable_{x}' for x in measurement_df.observableId.values]
        if len(set(measurement_observables) - set(observables.keys())):
            print(set(measurement_df.observableId.values) - set(observables.keys()))
            raise AssertionError('Unknown observables in measurement file')

    sbml_importer.sbml2amici(modelName=model_name,
                             output_dir=model_output_dir,
                             observables=observables,
                             constantParameters=constant_parameters,
                             sigmas=sigmas,
                             **kwargs
                             )

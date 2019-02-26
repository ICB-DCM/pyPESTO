"""
define your prior

"""
import numpy as np
from LogicleScale import (logicleTransform,
                          logicleInverseTransform,
                          logicleGradient,
                          logicleInverseGradient)

class Prior():

    def __init__(self,
                 priorType_list=None,
                 priorParameters_list=None,
                 estimate_list=None,
                 scale_list=None):

        # priorType_list = ['norm','lap','norm',...]
        # priorParameters_list = [[mean,cov],[mean,cov],...,[mean,cov]]
        # estimate_list: array of 0,1, prior works only on estimate
        # parameter array([0,1,1,0,0,...])
        # scale_list: list of parameter scales from model
        # ['lin','log10','logE','logicle',...]

        self.priorType_list = priorType_list
        self.priorParameters_list = priorParameters_list
        self.estimate_list = estimate_list
        self.scale_list = scale_list

    @property
    def has_priorType_list(self):
        return self.priorType_list is not None

    @property
    def has_estimate_list(self):
        return self.estimate_list is not None

    @property
    def has_scale_list(self):
        return self.scale_list is not None

    # define the prior
    def __call__(self, x, sensi_orders):

        # which parameter get penalized
        if self.has_estimate_list:
            estimate = np.where(self.estimate_list == 1)
        else:
            # default: estimate all parameters
            estimate = range(len(x))

        # get scale of all parameters
        if self.has_scale_list:

            # only works with arrays
            log10_index = [i for i, j in enumerate(self.scale_list)
                           if j == 'log10']
            x[log10_index] = 10**x[log10_index]

            log_index = [i for i, j in enumerate(self.scale_list)
                         if j == 'log']
            x[log_index] = np.exp(x[log_index])

            logE_index = [i for i, j in enumerate(self.scale_list)
                          if j == 'logE']
            x[logE_index] = 10**x[logE_index]-1

            T=100
            W=1
            M=4
            A=-W

            logicle_index = [i for i, j in enumerate(self.scale_list)
                           if j == 'logicle']
            x[logicle_index] = logicleInverseTransform(x[logicle_index],T,W,M,A)

        else:
            # default: all parameters are in linear scale
            log10_index = None
            log_index = None
            logE_index = None
            logicle_index = None

        # TODO
        # Hessian

        norm_index_1 = [i_par for i_par,
                        j_par in enumerate(self.priorType_list)
                        if j_par == 'norm']
        lap_index_1 = [i_par for i_par,
                       j_par in enumerate(self.priorType_list)
                       if j_par == 'lap']

        norm_index = np.intersect1d(norm_index_1, estimate)
        lap_index = np.intersect1d(lap_index_1, estimate)

        # LOGARITHMIC NORMAL PRIOR
        fun_norm = 0
        if len(norm_index) > 0:
            mean = np.array([self.priorParameters_list[i_par]
                         for i_par in norm_index])[:,0]
            cov = np.array([self.priorParameters_list[i_par]
                            for i_par in norm_index])[:,1]

            norm_log = -0.5*np.log(2*np.pi*cov**2) - (x[norm_index]-mean)**2/(2*cov**2)

            fun_norm = sum(norm_log)

            grad_norm = norm_log * (mean-x[norm_index])/cov**2

        # LOGARITHMIC LAPLACE PRIOR
        fun_lap = 0
        if len(lap_index) > 0:
            loc = np.array([self.priorParameters_list[i_par]
                            for i_par in lap_index])[:,0]

            scale = np.array([self.priorParameters_list[i_par]
                            for i_par in lap_index])[:,1]

            fun_lap = sum(- 1 / scale * np.abs(x[lap_index] - loc))#+ np.log(1 / (2 * scale)))

            grad_lap = np.sign(loc - x[lap_index]) / scale

        # calculate prior function
        prior_fun = fun_norm + fun_lap

        # calculate prior gradient
        prior_grad = np.zeros(len(x))

        if norm_index != []:
            prior_grad[norm_index] += grad_norm

        if lap_index != []:
            prior_grad[lap_index] += grad_lap

        # compute chainrule
        chainrule = np.zeros(len(x))
        chainrule[estimate] += 1

        #reset the different parameterizations and compute chainrule
        if log10_index is not None:
            # reset the values
            x[log10_index] = np.log10(x[log10_index])
            chainrule[log10_index] *= 10**x[log10_index] * np.log(10)

        if log_index is not None:
            # reset the values
            x[log_index] = np.log(x[log_index])
            chainrule[log_index] *= np.log(x[log_index])

        if logE_index is not None:
            # reset the values
            x[logE_index] = np.log10(x[logE_index] + 1)
            chainrule[logE_index] *= 10**x[logE_index]*np.log(10)

        if logicle_index is not None:
            #reset the values
            x[logicle_index] = logicleTransform(x[logicle_index],T,W,M,A)
            chainrule[logicle_index] *= logicleInverseGradient(x[logicle_index],T,W,M,A)

        # multiply the gradient by the chainrule
        prior_grad *= chainrule

        return {'prior_fun': prior_fun,
                'prior_grad': prior_grad,
                # 'prior_hess': hess,
                'chainrule': chainrule}

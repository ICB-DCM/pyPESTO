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
                 scale_list=None,
                 logicle_object=None,
                 prior=None):

        # priorType_list = ['norm','lap','norm',...]
        # priorParameters_list = [[mean,cov],[mean,cov],...,[mean,cov]]
        # estimate_list: [0,1,1,0,0,...] default: [1,...,1]
        # scale_list: ['lin', 'log10','log','logE','logicle',...] default: ['lin',...,'lin']

        self.priorType_list = priorType_list
        self.priorParameters_list = priorParameters_list
        self.estimate_list = estimate_list
        self.scale_list = scale_list
        self.logicle_object = logicle_object
        self.prior = prior

        # if self.has_prior:
        #     self.lap_index = self.prior.lap_index
        #     self.norm_index = self.prior.norm_index
        #     self.log10_index = self.prior.log10_index
        #     self.log_index = self.prior.log_index
        #     self.logE_index = self.prior.logE_index
        #     self.logicle_index = self.prior.logicle_index
        #     self.estimate = self.prior.estimate
        #     self.loc = self.prior.loc
        #     self.scale = self.prior.scale
        #     self.cov = self.prior.cov
        #     self.mean = self.prior.mean
        #
        # else:

        # which parameter get penalized
        if not self.has_estimate_list:
            # default: estimate all parameters
            estimate_list = np.ones(len(priorType_list))

        if isinstance(estimate_list, list):
            estimate_list = np.array(estimate_list)

        estimate = np.where(estimate_list == 1)

        # get scale of all parameters
        if self.has_scale_list:

            log10_index = get_index(self.scale_list, 'log10')

            log_index = get_index(self.scale_list, 'log')

            # log(1+x)
            logE_index = get_index(self.scale_list, 'logE')

            logicle_index = get_index(self.scale_list, 'logicle')

        else:
            # default: all parameters are in linear scale
            log10_index = []
            log_index = []
            logE_index = []
            logicle_index = []

        # TODO
        # Hessian

        # get indices of the different prior types
        norm_index_1 = get_index(self.priorType_list, 'norm')
        lap_index_1 = get_index(self.priorType_list, 'lap')

        # intersect indices of estimate parameter and
        # prior type indices
        norm_index = np.intersect1d(norm_index_1, estimate)
        lap_index = np.intersect1d(lap_index_1, estimate)

        loc = []
        scale = []
        if lap_index != []:
            loc = np.array([priorParameters_list[i_par]
                                 for i_par in lap_index])[:,0]
            scale = np.array([priorParameters_list[i_par]
                                   for i_par in lap_index])[:,1]

        mean = []
        cov = []
        if norm_index != []:
            mean = np.array([priorParameters_list[i_par]
                         for i_par in norm_index])[:,0]
            cov = np.array([priorParameters_list[i_par]
                            for i_par in norm_index])[:,1]

        self.log10_index = log10_index
        self.log_index = log_index
        self.logE_index = logE_index
        self.logicle_index = logicle_index
        self.estimate = estimate
        self.loc = loc
        self.scale = scale
        self.mean = mean
        self.cov = cov
        self.lap_index = lap_index
        self.norm_index = norm_index


#______________________________________________________________________

    @property
    def has_priorType_list(self):
        return self.priorType_list is not None

    @property
    def has_estimate_list(self):
        return self.estimate_list is not None

    @property
    def has_scale_list(self):
        return self.scale_list is not None

    @property
    def has_prior(self):
        return self.prior is not None

    def get_lap_index(self):
        return self.lap_index

    # calculate the prior
    def __call__(self, x, sensi_orders):

        # if input is of type list convert it to an
        # array for easier calculation
        if isinstance(x, list):
            x = np.array(x)

        # print('vor', x)
        if self.log10_index != []:
            x[self.log10_index] = 10**x[self.log10_index]

        if self.log_index != []:
            x[self.log_index] = np.exp(x[self.log_index])

        # log(1+x)
        if self.logE_index != []:
            x[self.logE_index] = 10**x[self.logE_index]-1
        # print('vor prior', x)
        if self.logicle_index != []:
            if np.isnan(x).any() == False:
                x[self.logicle_index] = logicleInverseTransform(x[self.logicle_index], self.logicle_object)

        # print('nach prior', x)

        # LOGARITHMIC NORMAL PRIOR
        fun_norm = 0
        if len(self.norm_index) > 0:

            self.norm_log = -0.5*np.log(2*np.pi*self.cov**2) - (x[self.norm_index]-self.mean)**2/(2*self.cov**2)

            fun_norm = sum(self.norm_log)

            grad_norm = self.norm_log * (self.mean-x[self.norm_index])/self.cov**2

        # LOGARITHMIC LAPLACE PRIOR
        fun_lap = 0
        if self.lap_index != []:

            fun_lap = sum(-1/self.scale * np.abs(x[self.lap_index]-self.loc) - np.log(2*self.scale))
            # print(np.log(2*self.scale))
            grad_lap = np.sign(self.loc - x[self.lap_index])/self.scale

        # calculate prior function
        prior_fun = fun_norm + fun_lap

        # calculate prior gradient
        prior_grad = np.zeros(len(x))

        if self.norm_index != []:
            prior_grad[self.norm_index] += grad_norm

        if self.lap_index != []:
            prior_grad[self.lap_index] += grad_lap

        # compute chainrule
        chainrule = np.zeros(len(x))
        chainrule[self.estimate] += 1

        #reset the different parameterizations and compute chainrule
        if self.log10_index is not None:
            # reset the values
            x[self.log10_index] = np.log10(x[self.log10_index])
            chainrule[self.log10_index] *= 10**x[self.log10_index] * np.log(10)

        if self.log_index is not None:
            # reset the values
            x[self.log_index] = np.log(x[self.log_index])
            chainrule[self.log_index] *= np.exp(x[self.log_index])

        if self.logE_index is not None:
            # reset the values
            x[self.logE_index] = np.log10(x[self.logE_index] + 1)
            chainrule[self.logE_index] *= 10**x[self.logE_index]*np.log(10)

        if self.logicle_index != [] and np.isnan(x).any() == False:
            #reset the values
            T = self.logicle_object.get_T()
            W = self.logicle_object.get_W()
            M = self.logicle_object.get_M()
            A = self.logicle_object.get_A()

            x[self.logicle_index] = logicleTransform(x[self.logicle_index], T=T, W=W, M=M, A=A)[0]
            chainrule[self.logicle_index] *= logicleInverseGradient(x[self.logicle_index], self.logicle_object)

        # multiply the gradient by the chainrule
        prior_grad *= chainrule

       # print('nach nach', x)
        return {'prior_fun': prior_fun,
                'prior_grad': prior_grad,
                # 'prior_hess': hess,
                'chainrule': chainrule}


def get_index(input_list, scale_name):
    return [i_par for i_par, j_par in enumerate(input_list) if j_par == scale_name]

#_____________________________________________________________________________________________________________
#
# class Prior():
#
#     def __init__(self,
#                  priorType_list=None,
#                  priorParameters_list=None,
#                  estimate_list=None,
#                  scale_list=None,
#                  logicle_object=None):
#
#         # priorType_list = ['norm','lap','norm',...]
#         # priorParameters_list = [[mean,cov],[mean,cov],...,[mean,cov]]
#         # estimate_list: array of 0,1, prior works only on estimate
#         # parameter array([0,1,1,0,0,...])
#         # scale_list: list of parameter scales from model
#         # ['lin','log10','logE','logicle',...]
#
#         self.priorType_list = priorType_list
#         self.priorParameters_list = priorParameters_list
#         self.estimate_list = estimate_list
#         self.scale_list = scale_list
#         self.logicle_object = logicle_object
#
#     @property
#     def has_priorType_list(self):
#         return self.priorType_list is not None
#
#     @property
#     def has_estimate_list(self):
#         return self.estimate_list is not None
#
#     @property
#     def has_scale_list(self):
#         return self.scale_list is not None
#
#     # calculate the prior
#     def __call__(self, x, sensi_orders):
#
#         # if input is of type list convert it to an
#         # array for easier calculation
#         if isinstance(x, list):
#             x = np.array(x)
#
#         # which parameter get penalized
#         if self.has_estimate_list:
#
#             if isinstance(self.estimate_list, list):
#                 self.estimate_list = np.array(self.estimate_list)
#
#             estimate = np.where(self.estimate_list == 1)
#         else:
#             # default: estimate all parameters
#             estimate = range(len(x))
#
#         # get scale of all parameters and
#         # transform them
#         if self.has_scale_list:
#
#             log10_index = [i_par for i_par, j_par in enumerate(self.scale_list)
#                            if j_par == 'log10']
#             x[log10_index] = 10**x[log10_index]
#
#             log_index = [i_par for i_par, j_par in enumerate(self.scale_list)
#                            if j_par == 'log']
#             x[log_index] = np.exp(x[log_index])
#
#             # log(1+x)
#             logE_index = [i_par for i_par, j_par in enumerate(self.scale_list)
#                            if j_par == 'logE']
#             x[logE_index] = 10**x[logE_index]-1
#
#             logicle_index = [i_par for i_par, j_par in enumerate(self.scale_list)
#                            if j_par == 'logicle']
#             if logicle_index != []:
#                 x[logicle_index] = logicleInverseTransform(x[logicle_index], self.logicle_object)
#
#         else:
#             # default: all parameters are in linear scale
#             log10_index = None
#             log_index = None
#             logE_index = None
#             logicle_index = None
#
#         # TODO
#         # Hessian
#
#         # get indices of the different prior types
#         norm_index_1 = [i_par for i_par,
#                         j_par in enumerate(self.priorType_list)
#                         if j_par == 'norm']
#         lap_index_1 = [i_par for i_par,
#                        j_par in enumerate(self.priorType_list)
#                        if j_par == 'lap']
#
#         # intersect indices of estimate parameter and
#         # prior type indices
#         norm_index = np.intersect1d(norm_index_1, estimate)
#         lap_index = np.intersect1d(lap_index_1, estimate)
#
#         # LOGARITHMIC NORMAL PRIOR
#         fun_norm = 0
#         if len(norm_index) > 0:
#             mean = np.array([self.priorParameters_list[i_par]
#                              for i_par in norm_index])[:, 0]
#             cov = np.array([self.priorParameters_list[i_par]
#                             for i_par in norm_index])[:, 1]
#
#             norm_log = -0.5 * np.log(2 * np.pi * cov ** 2) - (x[norm_index] - mean) ** 2 / (2 * cov ** 2)
#
#             fun_norm = sum(norm_log)
#
#             grad_norm = norm_log * (mean - x[norm_index]) / cov ** 2
#
#         # LOGARITHMIC LAPLACE PRIOR
#         fun_lap = 0
#         if len(lap_index) > 0:
#             loc = np.array([self.priorParameters_list[i_par]
#                             for i_par in lap_index])[:, 0]
#
#             scale = np.array([self.priorParameters_list[i_par]
#                               for i_par in lap_index])[:, 1]
#
#             fun_lap = sum(-1 / scale * np.abs(x[lap_index] - loc) - np.log(2 * scale))
#
#             grad_lap = np.sign(loc - x[lap_index]) / scale
#
#         # calculate prior function
#         prior_fun = fun_norm + fun_lap
#
#         # calculate prior gradient
#         prior_grad = np.zeros(len(x))
#
#         if norm_index != []:
#             prior_grad[norm_index] += grad_norm
#
#         if lap_index != []:
#             prior_grad[lap_index] += grad_lap
#
#         # compute chainrule
#         chainrule = np.zeros(len(x))
#         chainrule[estimate] += 1
#
#         # reset the different parameterizations and compute chainrule
#         if log10_index is not None:
#             # reset the values
#             x[log10_index] = np.log10(x[log10_index])
#             chainrule[log10_index] *= 10 ** x[log10_index] * np.log(10)
#
#         if log_index is not None:
#             # reset the values
#             x[log_index] = np.log(x[log_index])
#             chainrule[log_index] *= np.exp(x[log_index])
#
#         if logE_index is not None:
#             # reset the values
#             x[logE_index] = np.log10(x[logE_index] + 1)
#             chainrule[logE_index] *= 10 ** x[logE_index] * np.log(10)
#
#         if logicle_index != []:
#             # reset the values
#             T = self.logicle_object.get_T()
#             W = self.logicle_object.get_W()
#             M = self.logicle_object.get_M()
#             A = self.logicle_object.get_A()
#
#             np.array(x)[logicle_index] = logicleTransform(np.array(x)[logicle_index], T=T, W=W, M=M, A=A)[0]
#             chainrule[logicle_index] *= logicleInverseGradient(np.array(x)[logicle_index], self.logicle_object)
#
#         # multiply the gradient by the chainrule
#         prior_grad *= chainrule
#
#         return {'prior_fun': prior_fun,
#                 'prior_grad': prior_grad,
#                 # 'prior_hess': hess,
#                 'chainrule': chainrule}

#__________________________________________________________________

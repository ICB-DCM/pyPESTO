"""
define your prior

    INPUTS
        # priorType_list = ['norm','lap','norm',...]
        #
        # priorParameters_list = [[mean,cov],[mean,cov],...,[mean,cov]]
        # estimate_list: [0,1,1,0,0,...] default: [1,...,1]
        # scale_list: ['lin', 'log10','log','logE','logicle',...] default: ['lin',...,'lin']


"""

import numpy as np
from LogicleScale import (logicleTransform,
                          logicleInverseTransform,
                          logicleInverseGradient)


class Prior:

    def __init__(self,
                 priorType_list,
                 priorParameters_list,
                 estimate_list=None,
                 scale_list=None,
                 logicle_object=None,
                 shift_par=None,
                 prior=None):

        """
            INPUTS

            priorType_list = ['norm','lap','norm',...]
                with lap = laplacian prior, norm = normal prior

            priorParameters_list = [[mean,cov],[mean,cov],...,[mean,cov]]
                mean = the mean of the prior
                cov = the coveriance of the prior (= scale for the laplacian prior)

            estimate_list: [0,1,1,0,0,...] containing 0 and 1. 1 means that this
                parameter gets estimated, means prior information is added. For example
                for noise parameter we set the index to 0, as we do not want to penalize
                noise
                default: [1,...,1]

            scale_list: ['lin', 'log10','log','logE','logicle',...]
                lin = linear,
                log10 = logarithmic with base 10,
                log = logarithmic with base exp
                logE = log(x+epsilon): for this scale you have to specify the
                    input parameter shift_par = epsilon > 0, which specifies the defined
                    shift.
                logicle = logicle scale: for this scale you have to specify also the
                    needed scale parameter, which means you have to add the input
                    logicle_object.
                default: ['lin',...,'lin']

        """

        self.priorType_list = priorType_list
        self.priorParameters_list = priorParameters_list
        self.estimate_list = estimate_list
        self.scale_list = scale_list
        self.logicle_object = logicle_object
        self.shift_par = shift_par
        self.prior = prior

        # which parameter get penalized
        if not self.has_estimate_list:

            # default: estimate all parameters
            estimate_list = np.ones(len(priorType_list))

        if isinstance(estimate_list, list):
                estimate_list = np.array(estimate_list)

        # get indices of parameters which get estimated (=1)
        estimate = np.where(estimate_list == 1)

        # get scale of all parameters
        if self.has_scale_list:

            log10_index = get_index(self.scale_list, 'log10')

            log_index = get_index(self.scale_list, 'log')

            logE_index = get_index(self.scale_list, 'logE')  # log(1+x)

            logicle_index = get_index(self.scale_list, 'logicle')

        else:
            # default: all parameters are in linear scale
            log10_index = []
            log_index = []
            logE_index = []
            logicle_index = []

        # get indices of the different prior types
        norm_index_tmp = get_index(self.priorType_list, 'norm')
        lap_index_tmp = get_index(self.priorType_list, 'lap')

        # intersect indices of estimate parameter and
        # prior type indices
        norm_index = np.intersect1d(norm_index_tmp, estimate)
        lap_index = np.intersect1d(lap_index_tmp, estimate)

        # get scale parameters of the laplacian prior
        # and save them to list
        loc = []
        scale = []
        if lap_index != []:
            loc = np.array([priorParameters_list[i_par]
                            for i_par in lap_index])[:, 0]
            scale = np.array([priorParameters_list[i_par]
                              for i_par in lap_index])[:, 1]

        # get scale parameters of the normal prior
        # and save them to list
        mean = []
        cov = []
        if norm_index != []:
            mean = np.array([priorParameters_list[i_par]
                             for i_par in norm_index])[:, 0]
            cov = np.array([priorParameters_list[i_par]
                            for i_par in norm_index])[:, 1]

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

    @property
    def has_priorType_list(self):
        return self.priorType_list is not None

    @property
    def has_estimate_list(self):
        return self.estimate_list is not None

    @property
    def has_scale_list(self):
        return self.scale_list is not None

    def get_lap_index(self):
        return self.lap_index

    def get_norm_index(self):
        return self.norm_index

    # calculate the prior at x
    # sensi_orders = 1: calculate gradient to
    # sensi_orders = 0: calculate only function

    def __call__(self, x, sensi_orders):

        # if input is of type list convert it to an
        # array for easier calculation
        if isinstance(x, list):
            x = np.array(x)

        # transform all inputs corresponding to there scale
        if self.log10_index != []:
            x[self.log10_index] = 10**x[self.log10_index]

        if self.log_index != []:
            x[self.log_index] = np.exp(x[self.log_index])

        # log(1+x)
        if self.logE_index != []:
            x[self.logE_index] = 10**x[self.logE_index]-self.shift_par

        if self.logicle_index != []:
            if np.isnan(x).any() is False:
                x[self.logicle_index] = logicleInverseTransform(x[self.logicle_index], self.logicle_object)

        # LOGARITHMIC NORMAL PRIOR
        fun_norm = 0
        if self.norm_index != []:

            norm_log = -0.5*np.log(2*np.pi*self.cov**2) - (x[self.norm_index]-self.mean)**2/(2*self.cov**2)

            fun_norm = sum(norm_log)

            grad_norm = norm_log * (self.mean-x[self.norm_index])/self.cov**2

        # LOGARITHMIC LAPLACE PRIOR
        fun_lap = 0
        if self.lap_index != []:

            fun_lap = sum(-1/self.scale * np.abs(x[self.lap_index]-self.loc) - np.log(2*self.scale))

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

        # reset the different parameterizations and compute chainrule
        if self.log10_index is not None:
            x[self.log10_index] = np.log10(x[self.log10_index])
            chainrule[self.log10_index] *= 10**x[self.log10_index] * np.log(10)

        if self.log_index is not None:
            x[self.log_index] = np.log(x[self.log_index])
            chainrule[self.log_index] *= np.exp(x[self.log_index])

        if self.logE_index is not None:
            x[self.logE_index] = np.log10(x[self.logE_index] + 1)
            chainrule[self.logE_index] *= 10**x[self.logE_index]*np.log(10)

        if self.logicle_index != [] and np.isnan(x).any() is False:
            T = self.logicle_object.get_T()
            W = self.logicle_object.get_W()
            M = self.logicle_object.get_M()
            A = self.logicle_object.get_A()

            x[self.logicle_index] = logicleTransform(x[self.logicle_index], T=T, W=W, M=M, A=A)[0]
            chainrule[self.logicle_index] *= logicleInverseGradient(x[self.logicle_index], self.logicle_object)

        # multiply the gradient by the chainrule
        prior_grad *= chainrule

        return {'prior_fun': prior_fun,
                'prior_grad': prior_grad}
                # 'prior_hess': hess}


def get_index(list, scale_name):
    return [i_par for i_par, j_par in enumerate(list) if j_par == scale_name]

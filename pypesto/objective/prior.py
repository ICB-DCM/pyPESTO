"""
define your prior

"""
import numpy as np


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
        # ['lin','log10','log',...]

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

        else:
            # default: all parameters are in linear scale
            log10_index = None
            log_index = None
            logE_index = None

        # TODO
        # check if all parameters are already zero
        # print(x)
        # if x.all()<1e-4: flag=1
        # Hessian

        norm_index_1 = [i_par for i_par,
                        j_par in enumerate(self.priorType_list)
                        if j_par == 'norm']
        lap_index_1 = [i_par for i_par,
                       j_par in enumerate(self.priorType_list)
                       if j_par == 'lap']

        norm_index = np.intersect1d(norm_index_1, estimate)
        lap_index = np.intersect1d(lap_index_1, estimate)

        from scipy.stats import multivariate_normal, laplace

        fun_norm = 0
        grad_norm = []
        for i_par in norm_index:
            mean = self.priorParameters_list[i_par][0]
            cov = self.priorParameters_list[i_par][1]

            fun_norm += np.log(multivariate_normal.pdf(x[i_par],
                                                       mean=mean,
                                                       cov=cov))

            grad_norm.append(fun_norm * (mean-x[i_par])/cov**2)

        fun_lap = 0
        grad_lap = []
        for i_par in lap_index:
            loc = self.priorParameters_list[i_par][0]
            scale = self.priorParameters_list[i_par][1]

            fun_lap += np.log(laplace.pdf(x[i_par], loc=loc, scale=scale))

            grad_lap.append(np.sign(loc-x[i_par])/scale)

        # calculate prior function and gradient
        prior_fun = fun_norm + fun_lap

        prior_grad = np.zeros(len(x))

        if norm_index != []:
            prior_grad[norm_index] += grad_norm
        if lap_index != []:
            prior_grad[lap_index] += grad_lap

        # compute chainrule
        chainrule = np.zeros(len(x))
        chainrule[estimate] += 1

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

        prior_grad *= chainrule

        return {'prior_fun': prior_fun,
                'prior_grad': prior_grad,
                # 'prior_hess': hess,
                'chainrule': chainrule}

"""
define your prior

"""
import numpy as np

class Prior():

    def __init__(self,
                 function = None,
                 gradient = None,
                 priorType_list = None,
                 priorParameters_list = None,
                 estimate_list = None,
                 scale_list = None):

        #priorType_list = ['norm','lap','norm',...]
        #priorParameters_list = [[mean,cov],[mean,cov],...,[mean,cov]]
        #estimate_list: list of 0,1, prior works only on estimate parameter [0,1,1,0,0,...]
        #scale_list: list of parameter scales from model ['lin','log10','log',...]
        #log_prior: boolean if the prior is in log

        self.function = function
        self.gradient = gradient
        self.priorType_list = priorType_list
        self.priorParameters_list = priorParameters_list
        self.estimate_list = estimate_list
        self.scale_list = scale_list

    @property
    def has_function(self):
        return callable(self.function)

    @property
    def has_gradient(self):
        return callable(self.gradient)

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
    def __call__(self,x,sensi_orders):

        #which parameter get penalized
        if self.has_estimate_list:
            estimate = np.where(self.estimate_list == 1)
        else:
            #default: estimate all parameters
            estimate = range(len(x))

        #get scale of all parameters
        if self.has_scale_list:

            # log10_index = np.where(self.scale_list=='log10') #only works with arrays
            log10_index = [i for i, j in enumerate(self.scale_list) if j=='log10']
            x[log10_index] = 10**x[log10_index]

            log_index = [i for i, j in enumerate(self.scale_list) if j=='log']
            x[log_index] = np.exp(x[log_index])

        else:
            #default: all parameters are in linear scale
            log10_index = None
            log_index = None

        #CASE 1: function and gradient given
        if self.has_function and self.has_gradient:

            #compute prior function only with the parameters to estimate
            fun  = self.function(x[estimate],sensi_orders)

            #compute prior gradient only for the parameters to estimate
            grad = np.zeros(len(x))
            grad[estimate] = self.gradient(x[estimate],sensi_orders)



        #CASE 2: priorType_list given
        if self.has_priorType_list:

            norm_index_1 = [i for i, j in enumerate(self.priorType_list) if j=='norm']
            lap_index_1  = [i for i, j in enumerate(self.priorType_list) if j=='lap']

            norm_index = np.intersect1d(norm_index_1,estimate)
            lap_index  = np.intersect1d(lap_index_1,estimate)
            #print('lap:i', lap_index)
            #print('norm:i', norm_index)
            from scipy.stats import multivariate_normal, laplace, norm

            fun_norm=0
            grad_norm=[]
            for i in norm_index:
                mean = self.priorParameters_list[i][0]
                cov  = self.priorParameters_list[i][1]

                f1 = multivariate_normal.pdf(x[i], mean=mean, cov=cov)
                fun_norm += np.log(f1)

                g1 = norm_fun * (mean-x[i])/cov**2
                grad_norm.append(g1)


            fun_lap = 0
            grad_lap = []
            for i in lap_index:
                loc = self.priorParameters_list[i][0]
                scale = self.priorParameters_list[i][1]

                f2 = laplace.pdf(x[i], loc=loc, scale=scale)
                fun_lap += np.log(f2) - np.log(1/(2*scale))

                g2 = np.sign(loc - x[i]) / scale
                grad_lap.append(g2)


            #calculate prior function and gradient
            fun = fun_norm+fun_lap

            grad = np.zeros(len(x))
            if norm_index != []: grad[norm_index] += grad_norm
            if lap_index != []:  grad[lap_index]  += grad_lap

        #compute chainrule
        chainrule = np.zeros(len(x))
        chainrule[estimate] += 1

        if log10_index is not None:
            chainrule[log10_index] *= x[log10_index] * np.log(10)
            # reset the values
            x[log10_index] = np.log10(x[log10_index])

        if log_index is not None:
            chainrule[log_index] *= np.log(x[log_index])
            # reset the values
            x[log_index] = np.log(x[log_index])

        grad *= chainrule

        return {'prior_fun': fun, 'prior_grad': grad, 'chainrule': chainrule}
























# class NormalPrior(Prior):
#
#     def __init__(self, mean, cov, estimate_list=None, scale_list=None):
#
#         from scipy.stats import multivariate_normal
#         norm_fun  = lambda x,sensi_orders: multivariate_normal.pdf(x, mean=mean, cov=cov)
#         norm_grad = lambda x,sensi_orders: norm_fun * (mean-x)/cov**2
#
#         self.norm_fun = norm_fun
#         self.estimate_list = estimate_list
#         self.scale_list = scale_list
#
#         super().__init__(function=norm_fun,
#                          gradient=norm_grad,
#                          estimate_list=estimate_list,
#                          scale_list=scale_list,
#                          log_prior=False)
#
#
# class LaplacePrior(Prior):
#     pass
#



#
# f = lambda x,s: x + 1
# g = lambda x,s: x + 2
# s = ['lin','lin','lin']
# l = np.array([1,1,0])
#
#
# prior1 = Prior(function=f,gradient=g,estimate_list=l,scale_list=s)
#
# print(prior1(np.array([0.2,4,5]),1))

# x = np.linspace(-1,1,10)
# p = ['norm','norm','norm','norm','norm','norm','norm','norm','norm','norm']
# p1 = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
# e = np.ones(10)
#
# prior1 = Prior(priorType_list=p,priorParameters_list=p1,log_prior=False,estimate_list=e)
#
# print(prior1(x,1))
#
# import scipy
# z =scipy.stats.multivariate_normal.pdf(x,mean=0,cov=1)
# norm_grad =  z * (0-x)/1**2
# print(sum(z))
# print(z)
# print(norm_grad)
#
# import matplotlib.pyplot as plt
#
# plt.plot(x,norm_grad)
# plt.show()


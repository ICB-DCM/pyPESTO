"""
with this class, I can define any prior to add on my objective function

"""



class Prior():

    def __init__(self,
                 function = None,
                 gradient = None,
                 sensi_orders = None,
                 penalization_index = None,
                 scale = None):

        self.function = function
        self.gradient = gradient
        self.sensi_orders = sensi_orders
        self.penalization_index = penalization_index
        self.scale = scale



    def __call__(self,x):

        if self.scale != None:
            z = self.scale.function(x[self.penalization_index])
            chainrule = self.scale.gradient(x[self.penalization_index])
        else:
            # default linear scale
            z = x[self.penalization_index]
            chainrule = 1

        fun  = self.function(z)
        grad = np.zeros(len(x))
        grad[self.penalization_index] = self.gradient(z) * chainrule

        return {'prior_fun': fun, 'prior_grad': grad}


"""
with this class, I can define a personal scale, defined by a function, its inverse, and the gradient.

"""

class Scale:

    def __init__(self, fun = None, inv_fun = None, grad = None, name = None):

        self.function = fun
        self.inv_function = inv_fun
        self.gradient = grad
        self.name = name

    def get_fval(self,x):
        return self.function(x)

    def get_inv_fval(self,x):
        return self.inv_function(x)

    def get_grad(self,x):
        return self.gradient(x)



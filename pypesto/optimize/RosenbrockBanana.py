import pypesto
import pypesto.visualize as visualize
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pypesto.optimize as optimize
import time


###### Define Objective and Function
# first type of objective
objective1 = pypesto.Objective(fun=sp.optimize.rosen,
                               grad=sp.optimize.rosen_der,
                               hess=sp.optimize.rosen_hess)

# second type of objective
def rosen2(x):
    return (sp.optimize.rosen(x),
            sp.optimize.rosen_der(x),
            sp.optimize.rosen_hess(x))
objective2 = pypesto.Objective(fun=rosen2, grad=True, hess=True)

dim_full = 10
lb = -5 * np.ones((dim_full, 1))
ub = 5 * np.ones((dim_full, 1))

problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub)
problem2 = pypesto.Problem(objective=objective2, lb=lb, ub=ub)

'''
###### 3D Illustration of the Rosenbrock function
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)
for j in range(0, x.shape[0]):
    for k in range(0, x.shape[1]):
        z[j,k] = objective1([x[j,k], y[j,k]], (0,))

fig = plt.figure()
fig.set_size_inches(*(14,10))
ax = plt.axes(projection='3d')
ax.plot_surface(X=x, Y=y, Z=z)
plt.xlabel('x axis')
plt.ylabel('y axis')
ax.set_title('cost function values')
'''

###### Opimization
# create different optimizers
optimizer_pyswarms = optimize.PyswarmsOptimizer()
optimizer_scipydiffevolopt = optimize.ScipyDifferentialEvolutionOptimizer()
optimizer_cmaes = optimize.CmaesOptimizer()
optimizer_pyswarm = optimize.PyswarmOptimizer()

# set number of starts
n_starts = 20

# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)

# run optimizations for different optimizers
start_pyswarms = time.time()
result1_pyswarms = optimize.minimize(problem=problem1, optimizer=optimizer_pyswarms,
    n_starts=n_starts, history_options=history_options)
end_pyswarms = time.time()

start_scipy = time.time()
result1_scipydiffevolopt = optimize.minimize(problem=problem1, optimizer=optimizer_scipydiffevolopt,
    n_starts=n_starts, history_options=history_options)
end_scipy = time.time()

start_cmaes = time.time()
result1_cmaes = optimize.minimize(problem=problem1, optimizer=optimizer_cmaes,
    n_starts=n_starts, history_options=history_options)
end_cmaes = time.time()

start_pyswarm = time.time()
result1_pyswarm = optimize.minimize(problem=problem1, optimizer=optimizer_pyswarm,
    n_starts=n_starts, history_options=history_options)
end_pyswarm = time.time()


#### print times
print('Pyswarms: ' + '{:5.3f}s'.format(end_pyswarms - start_pyswarms))
print('Scipy: ' + '{:5.3f}s'.format(end_scipy - start_scipy))
print('Cmaes: ' + '{:5.3f}s'.format(end_cmaes - start_cmaes))
print('Pysawrm: ' + '{:5.3f}s'.format(end_pyswarm - start_pyswarm))


###### Visualite and compare optimization results
# plot separated waterfalls
#visualize.waterfall(result1_pyswarms, size=(15,6))
#visualize.waterfall(result1_scipydiffevolopt, size=(15,6))
#visualize.waterfall(result1_cmaes, size=(15,6))
#visualize.waterfall(result1_pyswarm, size=(15,6))



# -----------additional plots, comparisons, second optimum-----------

# compariosn of multiple optimizers
# Visualize Waterfall
visz = visualize.waterfall([result1_pyswarms, result1_scipydiffevolopt, result1_cmaes, result1_pyswarm],
                    legends=['Pyswarms', 'Scipy_DiffEvol', 'CMA-ES', 'PySwarm'],
                    scale_y='lin',
                    colors=[(31/255, 120/255, 180/255, 0.5), (178/255, 223/255, 138/255, 0.5),
                            (51/255, 160/255, 44/255, 0.5), (166/255, 206/255, 227/255, 0.5)])
                    #colors=['#1f78b4', '#b2df8a', '#33a02c', '#a6cee3'])
# change position of the legend
box = visz.get_position()
visz.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
visz.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
#visz.axhline(y=0, xmin=0, xmax=19, color='black', linestyle='--', alpha=0.75)

# Visulaize Parameters
para = visualize.parameters([result1_pyswarms, result1_scipydiffevolopt, result1_cmaes, result1_pyswarm],
                     legends=['PySwarms', 'Scipy_DiffEvol', 'CMA-ES', 'PySwarm'],
                     balance_alpha=True,
                     colors=[(31/255, 120/255, 180/255, 0.5), (178/255, 223/255, 138/255, 0.5),
                             (51/255, 160/255, 44/255, 0.5), (166/255, 206/255, 227/255, 0.5)])
                     #colors=['#1f78b4', '#b2df8a', '#33a02c', '#a6cee3'])
# change position of the legend
box = para.get_position()
para.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
#para.axvline(x=1, ymin=0, ymax=20, color='black', linestyle='--', alpha=0.75)


'''
# more options
visz.set_xlabel('Ordered optimizer run', fontsize=20)
visz.set_ylabel('Functional value', fontsize=20)
visz.set_title('Waterfall plot', fontdict={'fontsize': 20})
visz.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
para.set_xlabel('Parameter value', fontsize=20)
para.set_ylabel('Parameter', fontsize=20)
para.set_title('Estimated parameters', fontdict={'fontsize': 20})
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
'''


# retrieve second optimum
all_x = result1_pyswarms.optimize_result.get_for_key('x')
all_fval = result1_pyswarms.optimize_result.get_for_key('fval')
x = all_x[19]
fval = all_fval[19]
print('Second optimum at: ' + str(fval))

# create a reference point from it
ref = {'x': x, 'fval': fval, 'color': [
    0.2, 0.4, 1., 1.], 'legend': 'second optimum'}
ref = visualize.create_references(ref)

# new waterfall plot with reference point for second optimum
visualize.waterfall(result1_pyswarms, size=(15,6),
                    scale_y='lin', y_limits=[-1, 101],
                    reference=ref, colors=[0., 0., 0., 1.])

df = result1_pyswarms.optimize_result.as_dataframe(
    ['fval', 'n_fval', 'n_grad', 'n_hess', 'n_res', 'n_sres', 'time'])
df.head()


###### Optimizer History
# plot one list of waterfalls
visualize.optimizer_history([result1_pyswarms, result1_cmaes, result1_pyswarm],
                            legends=['Pyswarms', 'CMA-ES', 'PySwarm'],
                            reference=ref)


### profiling is missing
a = 4

import numpy as np
import matplotlib.pyplot as plt

def plotCostTrajectory(costTrajectory, color=None, scaleToIteration=1, legend_loc='upper right'):
    """Plot the provided cost trajectory
    
    Arguments: 
    costTrajectory: ndarray(numIterations x numStarts): cost over iterations for all optimizations
    
    Returns:
    
    """

    for start in range(costTrajectory.shape[1]):
        y = costTrajectory[:,start]
        y = y[~np.isnan(y)]
        c = color[start] if color and len(color) > 1 else color
        plt.semilogy(y, c=c)
    
    # set viewport
    
    minCost = np.nanmin(costTrajectory)
    maxCost = np.nanmax(costTrajectory[scaleToIteration:,:])
    plt.ylim(ymin=(1 - np.sign(minCost) * 0.02) * minCost,
             ymax=(1 + np.sign(maxCost) * 0.02) * maxCost)
    
    plt.legend(['start %d'%i for i in range(costTrajectory.shape[1]) ], loc=legend_loc)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    

def plotWaterfall(finalCost):
    """Plot "waterfall plot" 
    
    Sorted scatter plot of optimization results.
    
    Arguments:
    finalCost: ndarray(numStarts) of final cost
    
    Returns:
    
    """ 
    
    # x axis should have integer labels
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    order = np.argsort(finalCost)
    plt.scatter(range(finalCost.size), finalCost[:, order])
    
    plt.xlabel('Sorted start index')
    plt.ylabel('Final cost')
    
    
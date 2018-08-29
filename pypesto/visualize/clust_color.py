from scipy import cluster
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

def get_clust(result_fval):

    """
        Cluster cost function values

        Parameters
        ----------

        result_fval: list of cost function values

        Returns
        -------

        clust: clusters of cost function values

        clustsize: size of clusters form 1 to number of clusters

        ind_clust: indices to reconstruct 'clust' from a list with 1:number of clusters
        """

    clust = cluster.hierarchy.fcluster(
        cluster.hierarchy.linkage(result_fval),
        0.1, criterion='distance')
    uclust, ind_clust = np.unique(clust, return_inverse=True)
    clustsize = np.zeros(len(uclust))
    for iclustsize in range(len(uclust)):
        clustsize[iclustsize] = sum(clust == uclust[iclustsize])

    return clust, clustsize, ind_clust

def assigncolor(result_fval):

    """
        Assign color to each cluster

        Parameters
        ----------

        result_fval: list of cost function values

        Returns
        -------

        Col: a list of RGB, one for each cost function value
        """

    clust, clustsize, ind_clust = get_clust(result_fval)
    vmax = max(clust) - sum(clustsize == 1)
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cm.ScalarMappable(norm=cNorm)
    uind_col = vmax*np.ones(len(clustsize))
    sum_col = 0
    for iclustsize in range(len(clustsize)):
        if clustsize[iclustsize] > 1:
            uind_col[iclustsize] = sum_col
            sum_col = sum_col+1

    ind_col = uind_col[ind_clust]
    Col = scalarMap.to_rgba(ind_col)

    return Col

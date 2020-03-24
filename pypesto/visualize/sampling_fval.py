import matplotlib.pyplot as plt

def sampling_fval(results, ax=None,
               size=None, reference=None, colors=None, legends=None,
               balance_alpha=True, start_indices=None):
    """
    Plot parameter values.

    Parameters
    ----------

    results: pypesto.Result or list
        Optimization result obtained by 'optimize.py' or list of those

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    colors: list, or RGBA, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    legends: list or str
        Labels for line plots, one label per result object

    balance_alpha: bool (optional)
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)

    start_indices: list or int
        list of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    def x_as_ndarray(result):
        """Creates numpy array of size (n_chain, n_sample, n_par).
        Returns ndarray of parameters and fval
        """
        n_chain = len(result['chains'])
        n_sample = len(result['chains'][0])
        n_par = len(result['chains'][0][0]['x'])

        # reserve space
        arr = np.zeros((n_chain, n_sample, n_par))
        arr_fval = np.zeros((n_sample, 1))

        # fill
        for i_chain, chain in enumerate(result['chains']):
            for i_sample, sample in enumerate(chain):
                arr[i_chain, i_sample, :] = sample['x']

        for i_sample in range(0, n_sample):
            arr_fval[i_sample] = result['chains'][0][i_sample]['fval']

        return arr, arr_fval
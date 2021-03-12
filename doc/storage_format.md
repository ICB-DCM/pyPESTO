# Storage

It is important to be able to store analysis results efficiently, easily
accessible, and portable across systems. For this aim, pyPESTO allows to
store results in efficient, portable
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) files. Further, optimization
trajectories can be stored using various backends, including HDF5 and CSV.

In the following, describe the file formats.
For detailed information on usage, consult the `doc/example/hdf5_storage.ipynb`
notebook, and the API documentation for the `pypesto.objective.history` and
`pypesto.storage` modules.


## pyPESTO Problem

```
+ /problem/
  - Attributes:
    - filled by objective.get_config()
    - ...

  - lb [float n_par]
  - ub [float n_par]
  - lb_full [float n_par_full]
  - ub_full [float n_par_full]
  - dim [int]
  - dim_full [int]
  - x_fixed_values [float (n_par_full-n_par)]
  - x_fixed_indices [int (n_par_full-n_par)]
  - x_free_indices [int n_par]
  - x_names [str n_par_full]
```

## Parameter estimation

### Parameter estimation settings

Parameter estimation settings are saved in `/optimization/settings`.

### Parameter estimation results

Parameter estimation results are saved in `/optimization/results/`.

#### Results per local optimization

Results of the `$n`'th multistart a saved in the format

```
+ /optimization/results/$n/
  - fval: [float]
      Objective function value of best iteration
  - x: [float n_par_full]
      Parameter set of best iteration
  - grad: [float n_par_full]
      Gradient of objective function at point x
  - hess: [float n_par_full x n_par_full]
      Hessian matrix of objective function at point x
  - n_fval: [int]
      Total number of objective function evaluations
  - n_grad: [int]
      Number of gradient evaluations
  - n_hess: [int]
      Number of Hessian evaluations
  - x0: [float n_par_full]
      Initial parameter set
  - fval0: [float]
      Objective function value at starting parameters
  - exitflag: [str] Some exit flag
  - time: [float] Execution time
  - message: [str] Some exit message
```

#### Trace per local optimization

When objective function call histories are saved to HDF5, they are under
`/optimization/results/$n/trace/`.

```
+ /optimization/results/$n/trace/
  - fval: [float n_iter]
      Objective function value of best iteration
  - x: [float n_iter x n_par_full]
      Parameter set of best iteration
  - grad: [float n_iter x n_par_full]
      Gradient of objective function at point x
  - hess: [float n_iter x n_par_full x n_par_full]
      Hessian matrix of objective function at point x
  - time: [float n_iter] Executition time
  - chi2: [float n_iter x ...]
  - schi2: [float n_iter x ...]
```

## Sampling


### Sampling results

Sampling results are saved in `/sampling/chains/`.
```
+ /sampling/chains/$n/
```

TODO

## Profiling

TODO

### Profiling results

TODO

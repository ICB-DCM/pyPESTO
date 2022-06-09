# Storage

It is important to be able to store analysis results efficiently, easily
accessible, and portable across systems. For this aim, pyPESTO allows to
store results in efficient, portable
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) files. Further, optimization
trajectories can be stored using various backends, including HDF5 and CSV.

In the following, describe the file formats.
For detailed information on usage, consult the `doc/example/hdf5_storage.ipynb`
and `doc/example/store.ipynb` notebook, and the API documentation for the
`pypesto.objective.history` and `pypesto.storage` modules.


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
  - x: [float n_par_full]
      Parameter set of best iteration
  - grad: [float n_par_full]
      Gradient of objective function at point x
  - x0: [float n_par_full]
      Initial parameter set
  - Atrributes:
      - fval: [float]
          Objective function value of best iteration
      - hess: [float n_par_full x n_par_full]
          Hessian matrix of objective function at point x
      - n_fval: [int]
          Total number of objective function evaluations
      - n_grad: [int]
          Number of gradient evaluations
      - n_hess: [int]
          Number of Hessian evaluations
      - fval0: [float]
          Objective function value at starting parameters
      - optimizer: [str]
          Basic information on the used optimizer
      - exitflag: [str] Some exit flag
      - time: [float] Execution time
      - message: [str] Some exit message
      - id: [str] The id of the optimization

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

Sampling results are saved in `/sampling/results/`.
```
+ /sampling/results/
  - betas [float n_chains]
  - trace_neglogpost [float n_chains x (n_samples+1)]
  - trace_neglogprior [float n_chains x (n_samples+1)]
  - trace_x [float n_chains x (n_samples+1) x n_par]
  - Attributes:
    - time
```

## Profiling

### Profiling results

Profiling results are saved in `/profiling/$profiling_id/`, where `profiling_id` indicates the number of profilings done.
```
+/profiling/profiling_id/
  - $parameter_index/
    - exitflag_path [float n_iter]
    - fval_path [float n_iter]
    - gradnorm_path [float n_iter]
    - ratio_path [float n_iter]
    - time_path [float n_iter]
    - x_path [float n_par x n_iter]
    - Attributes:
      - time_total
      - IsNone
      - n_fval
      - n_grad
      - n_hess
```

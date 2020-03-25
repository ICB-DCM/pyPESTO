# HDF5 data format

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
The history is saved under `/optimization/results/$n/trace/$i`
```
+ /optimization/results/$n/trace/$i
  - fval: [float]
      Objective function value of best iteration
  - x: [float n_par_full]
      Parameter set of best iteration
  - grad: Gradient of objective function at point x
  - hess: Hessian matrix of objective function at point x
  - time: ...
  - chi2: ...
  - schi2: ...
```

## Sampling


### Sampling results

Sampling results are saved in `/sampling/chains/`.
```
+ /sampling/chains/$n/
```


## Profiling


### Profiling results

TODO

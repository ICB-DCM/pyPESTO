#!/bin/bash

# Run notebooks
#  Arguments 1, 2 specify a part of the models to run.
#  If nothing is specified, all are run.

# Environment

export PYPESTO_MAX_N_STARTS=20
export PYPESTO_MAX_N_SAMPLES=1000

base_dir='doc/example'

# Split notebooks up to parallelize execution
# When adding notebooks, make sure the load is balanced.

# Various topics notebooks
nbs_1=(
  'amici_import.ipynb'
  'conversion_reaction.ipynb'
  'fixed_parameters.ipynb'
  'petab_import.ipynb'
  'prior_definition.ipynb'
  'rosenbrock.ipynb'
  'store.ipynb'
  'synthetic_data.ipynb'
  'hdf5_storage.ipynb'
)

# Sampling notebooks
nbs_2=(
  # 'sampler_study.ipynb'
  'sampling_diagnostics.ipynb'
  'model_selection.ipynb'
)

# All tested notebooks
nbs_all=("${nbs_1[@]}" "${nbs_2[@]}")

# Select which notebooks to run
if [ $# -eq 0 ]; then
  nbs=("${nbs_all[@]}")
elif [ $1 -eq 1 ]; then
  nbs=("${nbs_1[@]}")
elif [ $1 -eq 2 ]; then
  nbs=("${nbs_2[@]}")
else
  echo "Unexpected input: $1"
fi

run_notebook () {
  # Run a notebook
  tempfile=$(tempfile)
  jupyter nbconvert \
    --ExecutePreprocessor.timeout=-1 --debug --stdout --execute \
    --to markdown $@ &> $tempfile
    ret=$?
    if [[ $ret != 0 ]]; then
      cat $tempfile
      exit $ret
    fi
    rm $tempfile
}

# Check that all notebooks are covered
for nb in `ls $base_dir | grep -E ipynb`; do
  # Check if notebook is in our list
  missing=true
  for nb_cand in "${nbs_all[@]}"; do
    if [[ $nb == $nb_cand ]]; then
      missing=false
      continue
    fi
  done
  if $missing; then
    echo "Notebook $nb is not covered in tests."
  fi
done

# Run all notebooks
for nb in "${nbs[@]}"; do
  echo $base_dir/$nb
  time run_notebook $base_dir/$nb
done
exit 0

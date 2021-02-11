#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=Test
#SBATCH --partition=short
## use --exclusive to get the whole nodes exclusively for this job
#SBATCH --exclusive

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=log/%x-%j-%j.log
## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem-per-cpu=5800        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-08:00:00   # Total time needed for job: Days-Hours:Minutes

#SBATCH --ntasks=3   # 2 workers, 1 manager

## Load the relevant modules needed for the job
module purge
module load OpenMPI/4.0.1-GCC-8.3.0-2.32
module load Python/3.8.2-GCCcore-9.3.0
module load SWIG/4.0.1-GCCcore-8.3.0 HDF5/1.10.6-gompi-2020a OpenBLAS/0.3.9-GCC-9.3.0 CMake/3.16.4-GCCcore-9.3.0 Boost/1.71.0-gompi-2019b

source venv/bin/activate

## Run your program or script
mpiexec -np $SLURM_NTASKS python -m mpi4py.futures example_MPIPool.py

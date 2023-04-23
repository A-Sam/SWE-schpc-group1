#!/bin/bash

# SLURM
#SBATCH -J swe_original_icc_i22r07c05s
#SBATCH -o output_icc_i22r07c05s.log
#SBATCH -e error_icc_i22r07c05s.log
#!SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --time=00:03:00
#SBATCH --verbose
#SBATCH --mail-type=end
#SBATCH --mail-user=ahmed.fouad@tum.de
#SBATCH --partition=cm2_inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --threads-per-core=2
#SBATCH --cpus-per-task=1
#SBATCH --sockets-per-node=2

export SLURM_CPU_BIND=verbose

scontrol -dd show job $SLURM_JOB_ID > scontrol.out

#  TODO Intel
#  1 - original code no vectorization
#      ENABLE_WRITERS OFF
#      ENABLE_NETCDF OFF
#      ENABLE_VECTORIZATION OFF
#      ENABLE_OPENMP OFF
#  2 - original code WITH vectorization
#      ENABLE_WRITERS OFF
#      ENABLE_NETCDF OFF
#      ENABLE_VECTORIZATION ON
#      ENABLE_OPENMP OFF
# Folder name: cm2_inter_Release_intel
# TODO $ sbatch -v ../Tools/job_script/job.sh

advixe-cl --collect=roofline --project-dir=.  --search-dir src:r=../ ./SWE-MPI-Runner -x 500 -y 500


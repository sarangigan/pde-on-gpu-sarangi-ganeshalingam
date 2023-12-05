#!/bin/bash -l
#SBATCH --job-name="2Dporous"
#SBATCH --output=2Dporous.%j.o
#SBATCH --error=2Dporous.%j.e
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda

export MPICH_RDMA_ENABLED_CUDA=0
export IGG_CUDAAWARE_MPI=0

srun -n1 bash -c 'julia -O3 PorousConvection/scripts/porous_convection_xpu.jl'
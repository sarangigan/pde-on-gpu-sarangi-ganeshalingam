Switching to atp/3.14.5.
Switching to cray-libsci/20.09.1.
Switching to cray-mpich/7.7.18.
Switching to craype/2.7.10.
Switching to gcc/9.3.0.
Switching to modules/3.2.11.4.
Switching to perftools-base/21.09.0.
Switching to pmi/5.0.17.
┌ Warning: There are known codegen bugs on CUDA 11.4 and earlier for older GPUs like your Tesla P100-PCIE-16GB.
│ Please use CUDA 11.5 or later, or switch to a different device.
└ @ CUDA /scratch/snx3000/julia/class211/daint-gpu/packages/CUDA/35NC6/lib/cudadrv/state.jl:246
slurmstepd: error: *** STEP 49921152.0 ON nid03130 CANCELLED AT 2023-11-08T14:22:28 DUE TO TIME LIMIT ***
slurmstepd: error: *** JOB 49921152 ON nid03130 CANCELLED AT 2023-11-08T14:22:28 DUE TO TIME LIMIT ***
srun: Job step aborted: Waiting up to 92 seconds for job step to finish.

[3706] signal (15): Terminated
in expression starting at /scratch/snx3000/class211/lecture7/PorousConvection/scripts/porous_convection_xpu.jl:164
ioctl at /lib64/libc.so.6 (unknown line)
srun: got SIGCONT
srun: forcing job termination

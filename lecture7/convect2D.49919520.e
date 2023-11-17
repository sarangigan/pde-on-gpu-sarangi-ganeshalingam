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
ERROR: LoadError: No current plot/subplot
Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:35
 [2] current
   @ /scratch/snx3000/julia/class211/daint-gpu/packages/Plots/sxUvK/src/plot.jl:14 [inlined]
 [3] frame(anim::Animation)
   @ Plots /scratch/snx3000/julia/class211/daint-gpu/packages/Plots/sxUvK/src/animation.jl:24
 [4] macro expansion
   @ /scratch/snx3000/julia/class211/daint-gpu/packages/Plots/sxUvK/src/animation.jl:232 [inlined]
 [5] macro expansion
   @ /scratch/snx3000/class211/lecture7/PorousConvection/scripts/porous_convection_xpu.jl:160 [inlined]
 [6] macro expansion
   @ /scratch/snx3000/julia/class211/daint-gpu/packages/Plots/sxUvK/src/animation.jl:251 [inlined]
 [7] porous_convection_2D(; do_vis::Bool)
   @ Main /scratch/snx3000/class211/lecture7/PorousConvection/scripts/porous_convection_xpu.jl:95
 [8] top-level scope
   @ /scratch/snx3000/class211/lecture7/PorousConvection/scripts/porous_convection_xpu.jl:165
in expression starting at /scratch/snx3000/class211/lecture7/PorousConvection/scripts/porous_convection_xpu.jl:165
srun: error: nid02944: task 0: Exited with exit code 1
srun: launch/slurm: _step_signal: Terminating StepId=49919520.0

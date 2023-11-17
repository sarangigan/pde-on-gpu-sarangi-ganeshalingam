Switching to atp/3.14.5.
Switching to cray-libsci/20.09.1.
Switching to cray-mpich/7.7.18.
Switching to craype/2.7.10.
Switching to gcc/9.3.0.
Switching to modules/3.2.11.4.
Switching to perftools-base/21.09.0.
Switching to pmi/5.0.17.
ERROR: SystemError: opening file "/scratch/snx3000/class211/lecture7/PorousConvection/scripts/porousConvection_3D_xpu.jl": No such file or directory
Stacktrace:
  [1] systemerror(p::String, errno::Int32; extrainfo::Nothing)
    @ Base ./error.jl:176
  [2] #systemerror#82
    @ ./error.jl:175 [inlined]
  [3] systemerror
    @ ./error.jl:175 [inlined]
  [4] open(fname::String; lock::Bool, read::Nothing, write::Nothing, create::Nothing, truncate::Nothing, append::Nothing)
    @ Base ./iostream.jl:293
  [5] open
    @ ./iostream.jl:275 [inlined]
  [6] open(f::Base.var"#418#419"{String}, args::String; kwargs::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ Base ./io.jl:393
  [7] open
    @ ./io.jl:392 [inlined]
  [8] read
    @ ./io.jl:473 [inlined]
  [9] _include(mapexpr::Function, mod::Module, _path::String)
    @ Base ./loading.jl:1959
 [10] include(mod::Module, _path::String)
    @ Base ./Base.jl:457
 [11] exec_options(opts::Base.JLOptions)
    @ Base ./client.jl:307
 [12] _start()
    @ Base ./client.jl:522
srun: error: nid02333: task 0: Exited with exit code 1
srun: launch/slurm: _step_signal: Terminating StepId=49931556.0

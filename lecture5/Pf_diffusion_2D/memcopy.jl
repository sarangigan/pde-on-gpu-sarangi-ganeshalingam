using Plots, Plots.Measures, Printf
using BenchmarkTools

default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

# numerics
    
nx, ny  = 512, 512
nt      = 2e4
   
# array initialisation
C       = rand(Float64, nx, ny)
C2      = copy(C)
A       = copy(C)

function compute_ap(C2,C,A)
    #iteration loop
    C2 .= C .+ A
    #error
end

function compute_kp!(C2,C,A)
    Threads.@threads for i in 1:length(C)
        C2[i] = C[i] + A[i] 
        niter +=1
    end
end
function memcopy(bench)
    
    if bench == :loop
    # iteration loop
        t_tic = 0.0
        for iter=1:nt
            compute_kp!(C2,C,A)  
              
        end

        t_toc = Base.time() - t_tic
        print(t_toc)
        elseif bench == :btool
            t_toc = @belapsed compute_ap(C2,C,A)
    end
    
    #=
    t_toc = Base.time() - t_tic
    A_eff = 3*8*nx*ny
    t_it  = t_toc/niter #execution time per iteration   
    T_eff = A_eff/t_it  #effective memory throughput
   =#
end

memcopy(:loop)
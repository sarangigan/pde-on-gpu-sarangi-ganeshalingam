using Plots, Plots.Measures, Printf
using BenchmarkTools

default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

# numerics
nx = ny   = 512
nt        = 2e4
n         = 16 * 2 .^ (1:8)
n_iter    = 0
global T_eff_max = 0

# array initialisation
T_eff_btool_ap = Float64[]
T_eff_btool_kp = Float64[]
T_eff_loop = Float64[]

function compute_ap!(C2,C,A)

    C2 .= C .+ A

end

function compute_kp!(C2,C,A,nx,ny)
    Threads.@threads for iy in 1:ny
        for ix = 1:nx

            C2[ix, iy] = C[ix, iy] + A[ix, iy] 

        end
    end
end

function memcopy(bench, nx, ny)
    # array initialisation
    C       = rand(Float64, nx, ny)
    C2      = copy(C)
    A       = copy(C)

    if bench == :loop
        t_tic = 0.0; iter = 0

        for iter=1:nt
            if iter == 0.1*nx
                t_tic = Base.time()
            end

            compute_kp!(C2,C,A,nx,ny)

        end

        t_toc = (Base.time() - t_tic)
        A_eff = 3*8*nx*ny/1e9
        t_it  = t_toc/(nt-0.1*nx) #execution time per iteration   
        T_eff = A_eff/t_it  #effective memory throughput

        if T_eff > T_eff_max
            global T_eff_max = T_eff 
        end


    elseif bench == :btool_ap
        t_it = @belapsed compute_ap!($C2,$C,$A)
        A_eff = 3*8*nx*ny/1e9
        T_eff= A_eff/t_it


    elseif bench == :btool_kp
        t_it = @belapsed compute_kp!($C2,$C,$A,$nx,$ny)
        A_eff = 3*8*nx*ny/1e9
        T_eff= A_eff/t_it

    end
    return T_eff
   
end


for n_iter in 1:8
    nx_plot   = n[n_iter]
    ny_plot   = n[n_iter]
    T_eff_kp  = memcopy(:btool_kp, nx_plot, ny_plot)
    T_eff_ap  = memcopy(:btool_ap, nx_plot, ny_plot)
    T_eff_max_loop = memcopy(:loop, nx_plot, ny_plot)
    push!(T_eff_btool_kp, T_eff_kp)
    push!(T_eff_btool_ap, T_eff_ap)
    push!(T_eff_loop, T_eff_max_loop)
end
T_eff_loop_max_value = maximum(T_eff_loop)

plot(n, T_eff_btool_kp, label="Kernel Based Approach", marker=:circle)
plot!(n, T_eff_btool_ap, label="Array Based Approach", marker=:square)

xlabel!("Grid Size (nx × ny)")
ylabel!("Memory Throughput (GB/s)")
title!("Memory Throughput vs. Grid Size")
annotate!([(5, 0.3, text("Max. T_eff in loop-based approach = $T_eff_loop_max_value ", :left, 10, :red))])
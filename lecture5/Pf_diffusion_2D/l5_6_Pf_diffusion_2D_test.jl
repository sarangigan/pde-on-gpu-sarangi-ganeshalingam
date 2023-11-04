using Plots, Plots.Measures, Printf
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)
using BenchmarkTools
using Test

function compute!(Pf,qDx,qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx_β_dτ, _dy_β_dτ)
    compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    update_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
    return nothing
end

function compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    nx,ny=size(Pf)
    niter = 0
    Threads.@threads for iy=1:ny
        for ix=1:nx-1
            qDx[ix + 1, iy] -= (qDx[ix + 1, iy] + k_ηf_dx * (Pf[ix + 1, iy] - Pf[ix, iy])) * _1_θ_dτ
            #niter+=1
        end
        #niter +=1
    end

    Threads.@threads for iy=1:ny-1
        for ix=1:nx
            qDy[ix, iy + 1] -= (qDy[ix, iy + 1] + k_ηf_dy * (Pf[ix, iy + 1] - Pf[ix, iy])) * _1_θ_dτ
        end
    end
    return nothing
end
function update_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
    nx,ny=size(Pf)
    Threads.@threads for iy=1:ny
        for ix=1:nx
            Pf[ix, iy]     -= ((qDx[ix + 1, iy] - qDx[ix, iy]) * _dx_β_dτ + (qDy[ix, iy + 1] - qDy[ix, iy]) * _dy_β_dτ)
        end
    end
end

function Pf_diffusion_2D(;do_check = false, nx, ny)
    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics
    #nx, ny  = 511, 511
    ϵtol    = 1e-8
    #maxiter = max(nx, ny)
    ncheck  = ceil(Int, 0.25max(nx, ny))
    cfl     = 1.0 / sqrt(2.1)
    re      = 2π
    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    xtest = [5, Int(cld(0.6*lx, dx)), nx-10]
    ytest = Int(cld(0.5*ly, dy))
    # array initialisation
    Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
    qDx     = zeros(Float64, nx + 1, ny)
    qDy     = zeros(Float64, nx, ny + 1)
    r_Pf    = zeros(nx, ny)
    k_ηf_dx, k_ηf_dy = k_ηf / dx, k_ηf / dy
    _dx_β_dτ, _dy_β_dτ = 1 / dx / β_dτ, 1 / dy / β_dτ
    _1_θ_dτ = 1.0./(1.0 + θ_dτ)
    
    # iteration loop
        iter = 1; err_Pf = 2ϵtol; t_tic = 0.0; niter = 0
        while err_Pf >= ϵtol && iter <= maxiter
            (iter == 11) && (t_tic = Base.time(); niter = 0)

            t_toc = @belapsed compute!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx_β_dτ, _dy_β_dτ)

            if do_check && (iter % ncheck == 0)
                r_Pf .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
                err_Pf = maximum(abs.(r_Pf))
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Pf'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
            end
            iter += 1; niter += 1
        end

   
        return Pf, xtest, ytest
    end
    


@testset "Diffusion 2D Tests" begin
    nx_values = ny_values = 16 * 2 .^ (2:5) .- 1
    maxiter = 500

    Chk = [[0.00785398056115133 0.007853980637555755 0.007853978592411982],
           [0.00787296974549236 0.007849556884184108 0.007847181374079883],
           [0.00740912103848251 0.009143711648167267 0.007419533048751209],
           [0.00566813765849919 0.004348785338575644 0.005618691590498087]]
    

    for ires=1:4

        nx = nx_values[ires]
        ny = ny_values[ires]
        
        Pf,xtest_result, ytest_result = Pf_diffusion_2D(; nx, ny)
        @test Pf[xtest_result,ytest_result]' ≈ Chk[ires] atol=0.0005
        println(Pf[xtest_result,ytest_result]',  Chk[ires] )
    end

end

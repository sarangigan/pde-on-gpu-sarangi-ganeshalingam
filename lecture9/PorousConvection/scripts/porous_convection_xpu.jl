const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
end
using Printf, Plots


test = true

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])



"""
##compute_flux!

Computation of the darcy flux with the obtained fluxes, pressure and temperature of previous time step.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for qDx, qDy, Pf, k_ηf, _dx, _dy, _1_θ_dτ_D, αρgx, αρgy, T when calling the function.

###Syntax
@parallel compute_flux!(qDx, qDy, Pf, k_ηf, _dx, _dy, _1_θ_dτ_D, αρgx, αρgy, T)
"""
@parallel function compute_flux!(qDx, qDy, Pf, k_ηf, _dx, _dy, _1_θ_dτ_D, αρgx, αρgy, T)
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) * _dx - αρgx * @av_xa(T))) * _1_θ_dτ_D
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) * _dy - αρgy * @av_ya(T))) * _1_θ_dτ_D
    return nothing
end


"""
##update_Pf!
Computation of the pressure at new time step based on the updated darcy fluxes in x, y and z direction.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for Pf, qDx, qDy, _dx, _dy, _β_dτ_D when calling the function.

###Syntax
@parallel update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)
"""
@parallel function update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy) * _β_dτ_D 
    return nothing
end


"""
##compute_Tflux!
Computation of flux initiated by temperature difference.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for qTx, qTy, dTdt, λ_ρCp, qDx, qDy, T, T_old, _dx, _dy, _dt, _1_θ_dτ_T, _ϕ when calling the function.
 
###Syntax
@parallel compute_Tflux!(qTx, qTy, dTdt, λ_ρCp, qDx, qDy, T, T_old, _dx, _dy, _dt, _1_θ_dτ_T, _ϕ)
"""
@parallel_indices (ix, iy) function compute_Tflux!(qTx, qTy, dTdt, λ_ρCp, qDx, qDy, T, T_old, _dx, _dy, _dt, _1_θ_dτ_T, _ϕ)
    if (ix <= size(qTx, 1) && iy <= size(qTx, 2))
        qTx[ix,iy] = qTx[ix,iy] - (qTx[ix,iy] + λ_ρCp * (T[ix+1,iy+1]-T[ix,iy+1]) * _dx ) * _1_θ_dτ_T
    end

    if (ix <= size(qTy, 1) && iy <= size(qTy, 2))
        qTy[ix,iy] = qTy[ix,iy] - (qTy[ix,iy] + λ_ρCp * (T[ix+1,iy+1]-T[ix+1,iy]) * _dy ) * _1_θ_dτ_T
    end

    if (ix <= size(dTdt, 1) && iy <= size(dTdt, 2))
        dTdt[ix,iy] = (T[ix+1,iy+1] - T_old[ix+1,iy+1]) * _dt +
        (max(qDx[ix+1, iy+1], 0.0) * (T[ix+1, iy+1] - T[ix, iy+1]) * _dx +
        min(qDx[ix+2, iy+1], 0.0) * (T[ix+2, iy+1] - T[ix+1, iy+1]) * _dx +
        max(qDy[ix+1, iy+1], 0.0) * (T[ix+1, iy+1] - T[ix+1, iy]) * _dy +
        min(qDy[ix+1, iy+2], 0.0) * (T[ix+1, iy+2] - T[ix+1, iy+1]) * _dy) * _ϕ
    end

    return nothing
end


"""
##update_T

Calculation of T at the new time step after computing temperature based flux in x and y direction.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for qTx, qTy, λ_ρCp, T, _dx, _dy, _1_θ_dτ_T when calling the function.

##Syntax
@parallel update_T!(T, dTdt, qTx, _dx, qTy, _dy, _dt_β_dτ_T)
"""
@parallel function update_T!(T, dTdt, qTx, _dx, qTy, _dy, _dt_β_dτ_T)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy) * _dt_β_dτ_T
    return nothing
end


"""
##bc_x
Set the boundary cells in x direction A[1, iy, iz] and A[end, iy, iz] to be equal to the adjecent cells A[2, iy, iz] and A[end-1, iy, iz].
This function contains parallel computation and needs to be initialised with '@parallel_indices'.

Insert a matrix that should be updates when calling the function.

###Syntax
parallel_indices (iy, iz) bc_x!(A)
"""
@parallel_indices (iy) function bc_x!(A)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel function compute_r!(r_Pf, r_T, qDx, qDy, qTx, qTy, dTdt, _dx, _dy)
    @all(r_Pf) = @d_xa(qDx) * _dx + @d_ya(qDy) * _dy
    @all(r_T)  = @all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy
    return nothing
end

"""
##porous_convection_2D
Computating simulation of porous convection in 2D using a numerical method. Note that some tasks are conducted by using parallel computation.
This function returns the updated temperature and generates a visualisation.

###Syntax
without visualisation: porous_convection_2D()
with visualisation: porous_convection_2D(do_viz = true)
"""
@views function porous_convection_2D(; nx = 1023, ny = 511, nt = 4000, do_vis = false, do_check = false)
    # physics
    lx, ly     = 40.0, 20.0
    k_ηf       = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2)
    ΔT         = 200.0
    ϕ          = 0.1
    _ϕ         = 1.0 / ϕ
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    # ny         = 63
    # nx         = 2 * (ny + 1) - 1
    nx, ny     = nx, ny
    nt         = nt
    re_D       = 4π
    cfl        = 1.0 / sqrt(2.1)
    maxiter    = 10max(nx, ny)
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx, ny))
    # preprocessing
    dx, dy      = lx / nx, ly / ny
    _dx, _dy    = 1.0 / dx, 1.0 / dy
    xn, yn      = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly, 0, ny + 1)
    xc, yc      = av1(xn), av1(yn)
    θ_dτ_D      = max(lx, ly) / re_D / cfl / min(dx, dy)
    _1_θ_dτ_D   = 1.0 / (1.0 + θ_dτ_D)
    β_dτ_D      = (re_D * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    _β_dτ_D     = 1.0 / β_dτ_D
    # init
    Pf           = @zeros(nx, ny)
    r_Pf         = @zeros(nx, ny)
    qDx, qDy     = @zeros(nx + 1, ny), @zeros(nx, ny + 1)
    qDx_c, qDy_c = zeros(nx, ny), zeros(nx, ny)
    qDmag        = zeros(nx, ny)
    T            = Data.Array(@. ΔT * exp(-xc^2 - (yc' + ly / 2)^2))
    T[:, 1]      .= ΔT / 2
    T[:, end]    .= -ΔT / 2
    T_old        = copy(T)
    dTdt         = @zeros(nx - 2, ny - 2)
    r_T          = @zeros(nx - 2, ny - 2)
    qTx          = @zeros(nx - 1, ny - 2)
    qTy          = @zeros(nx - 2, ny - 1)
    # vis
    st     = ceil(Int, nx / 25)
    Xc, Yc = [x for x in xc, y in yc], [y for x in xc, y in yc]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    # action
    iframe = 0
    for it = 1:nt
        T_old .= T

        # time step
        dt = if it == 1
            0.1 * min(dx, dy) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1)
        end
        _dt = 1.0 / dt
        re_T        = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T      = max(lx, ly) / re_T / cfl / min(dx, dy)
        _1_θ_dτ_T   = 1.0 / (1.0 + θ_dτ_T)
        β_dτ_T      = (re_T * λ_ρCp) / (cfl * min(dx, dy) * max(lx, ly))
        _dt_β_dτ_T  = 1.0 / (1.0 / dt + β_dτ_T)
        # iteration loop
        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # hydro
            @parallel compute_flux!(qDx, qDy, Pf, k_ηf, _dx, _dy, _1_θ_dτ_D, αρgx, αρgy, T)
            @parallel update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)
            # thermo
            @parallel compute_Tflux!(qTx, qTy, dTdt, λ_ρCp, qDx, qDy, T, T_old, _dx, _dy, _dt, _1_θ_dτ_T, _ϕ)
            @parallel update_T!(T, dTdt, qTx, _dx, qTy, _dy, _dt_β_dτ_T)

            @parallel (1:size(T,2)) bc_x!(T)

            if (do_check) && (iter % ncheck == 0)
                @parallel compute_r!(r_Pf, r_T, qDx, qDy, qTx, qTy, dTdt, _dx, _dy)
                err_D = maximum(abs.(r_Pf))
                err_T = maximum(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx, err_D, err_T)
            end
            iter += 1
        end
        @printf("it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
        # visualisation
        if do_vis
            ENV["GKSwstype"] = "nul"
            if (isdir("viz2D_out") == false) mkdir("viz2D_out") end
            loadpath = "viz2D_out/"
            anim = Animation(loadpath, String[])
            if it % nvis == 0
                println("Animation directory: $(anim.dir)")
                qDx_c .= avx(Array(qDx))
                qDy_c .= avy(Array(qDy))
                qDmag .= sqrt.(qDx_c .^ 2 .+ qDy_c .^ 2)
                qDx_c ./= qDmag
                qDy_c ./= qDmag
                qDx_p = qDx_c[1:st:end, 1:st:end]
                qDy_p = qDy_c[1:st:end, 1:st:end]
                heatmap(xc, yc, Array(T)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo)
                quiver!(Xp[:], Yp[:]; quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black)
                savefig(@sprintf("viz2D_out/%04d.png", iframe))
            end
        end
        iframe += 1
    end
    return T
end

if test==false
    porous_convection_2D(do_vis=true,do_check=true)
else
end




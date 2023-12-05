const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end
using Printf, Plots

test = true

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

"""
##compute_flux!

Computation of the darcy flux with the obtained fluxes, pressure and temperature of previous time step.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρgx, αρgy, αρgz, T when calling the function.

###Syntax
@parallel compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρgx, αρgy, αρgz, T)
"""
@parallel function compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρgx, αρgy, αρgz, T)
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) * _dx - αρgx * @av_xa(T))) * _1_θ_dτ_D
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) * _dy - αρgy * @av_ya(T))) * _1_θ_dτ_D
    @inn_z(qDz) = @inn_z(qDz) - (@inn_z(qDz) + k_ηf * (@d_za(Pf) * _dz - αρgz * @av_za(T))) * _1_θ_dτ_D
    return nothing
end


"""
##update_Pf!
Computation of the pressure at new time step based on the updated darcy fluxes in x, y and z direction.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D when calling the function.

###Syntax
@parallel update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
"""
@parallel function update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz) * _β_dτ_D 
    return nothing
end

"""
##compute_Tflux!
Computation of flux initiated by temperature difference.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for qTx, qTy, qTz, qDx, qDy, qDz, dTdt, λ_ρCp, T, T_old, _dx, _dy, _dz, _dt,  _1_θ_dτ_T, _ϕ when calling the function.
 
###Syntax
@parallel compute_Tflux!(qTx, qTy, qTz, qDx, qDy, qDz, dTdt, λ_ρCp, T, T_old, _dx, _dy, _dz, _dt,  _1_θ_dτ_T, _ϕ)
"""
@parallel_indices (ix, iy, iz) function compute_Tflux!(qTx, qTy, qTz, qDx, qDy, qDz, dTdt, λ_ρCp, T, T_old, _dx, _dy, _dz, _dt,  _1_θ_dτ_T, _ϕ)
    if (ix <= size(qTx, 1) && iy <= size(qTx, 2) && iz <= size(qTx, 3))
        qTx[ix,iy,iz] = qTx[ix,iy,iz] - (qTx[ix,iy,iz] + λ_ρCp * (T[ix+1,iy+1,iz+1]-T[ix,iy+1,iz+1]) * _dx ) * _1_θ_dτ_T
    end
    if (ix <= size(qTy, 1) && iy <= size(qTy, 2) && iz <= size(qTy, 3))
        qTy[ix,iy,iz] = qTy[ix,iy,iz] - (qTy[ix,iy,iz] + λ_ρCp * (T[ix+1,iy+1,iz+1]-T[ix+1,iy,iz+1]) * _dy ) * _1_θ_dτ_T
    end
    if (ix <= size(qTz, 1) && iy <= size(qTz, 2) && iz <= size(qTz, 3))
        qTz[ix,iy,iz] = qTz[ix,iy,iz] - (qTz[ix,iy,iz] + λ_ρCp * (T[ix+1,iy+1,iz+1]-T[ix+1,iy+1,iz]) * _dz ) * _1_θ_dτ_T
    end
    if (ix <= size(dTdt, 1) && iy <= size(dTdt, 2) && iz <= size(dTdt, 3))
        dTdt[ix, iy, iz] = (T[ix+1, iy+1, iz+1] - T_old[ix+1, iy+1, iz+1]) * _dt +
                           (max(qDx[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix, iy+1, iz+1]) * _dx +
                            min(qDx[ix+2, iy+1, iz+1], 0.0) * (T[ix+2, iy+1, iz+1] - T[ix+1, iy+1, iz+1]) * _dx +
                            max(qDy[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy, iz+1]) * _dy +
                            min(qDy[ix+1, iy+2, iz+1], 0.0) * (T[ix+1, iy+2, iz+1] - T[ix+1, iy+1, iz+1]) * _dy +
                            max(qDz[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy+1, iz]) * _dz +
                            min(qDz[ix+1, iy+1, iz+2], 0.0) * (T[ix+1, iy+1, iz+2] - T[ix+1, iy+1, iz+1]) * _dz) * _ϕ
    end
    return nothing 
end


"""
##update_T

Calculation of T at the new time step after computing temperature based flux in x and y direction.
This function contains parallel computation and needs to be initialised with '@parallel'.

Insert values for T, dTdt, qTx, _dx, qTy, _dy, qTz, _dz, _dt_β_dτ_T when calling the function.

##Syntax
@parallel update_T!(T, dTdt, qTx, _dx, qTy, _dy, qTz, _dz, _dt_β_dτ_T)
"""
@parallel function update_T!(T, dTdt, qTx, _dx, qTy, _dy, qTz, _dz, _dt_β_dτ_T)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz) * _dt_β_dτ_T
    return nothing
end

"""
##bc_x
Set the boundary cells in x direction A[1 , iy, iz] and A[end, iy, iz] to be equal to the adjecent cells A[2 , iy, iz] and A[end-1, iy, iz].
This function contains parallel computation and needs to be initialised with '@parallel_indices'.
Insert a matrix that should be updates when calling the function.

###Syntax
@parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
"""
@parallel_indices (iy, iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

"""
##bc_y
Set the boundary cells in x direction A[ix , 1, iz] and A[ix, end, iz] to be equal to the adjecent cells A[ix, 2, iz] and A[ix, end-1, iz].
This function contains parallel computation and needs to be initialised with '@parallel_indices'.
Insert a matrix that should be updates when calling the function.

###Syntax
@parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)
"""
@parallel_indices (ix, iz) function bc_y!(A)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end


@parallel function compute_r!(r_Pf, r_T, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz)
    @all(r_Pf) = @d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz
    @all(r_T)  = @all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz
    return nothing
end

"""
##porous_convection_3D
Computating simulation of porous convection in 3D using a numerical method. Note that some tasks are conducted by using parallel computation.
This function returns the updated temperature and generates a visualisation.

###Syntax
without visualisation: porous_convection_3D()
with visualisation: porous_convection_3D(do_viz = true)
"""
@views function porous_convection_3D(; nx = 255, ny = 127, nz = 127, nt = 2000, do_viz = false, do_check = false)
    # physics
    lx, ly, lz       = 40.0, 20.0, 20.0
    Ra               = 1000
    k_ηf             = 1.0
    αρgx, αρgy, αρgz = 0.0, 0.0, 1.0
    αρg              = sqrt(αρgx^2 + αρgy^2 + αρgz^2)
    ΔT               = 200.0
    ϕ                = 0.1
    λ_ρCp            = 1 / Ra * (αρg * k_ηf * ΔT * lz / ϕ) # Ra = αρg*k_ηf*ΔT*lz/λ_ρCp/ϕ
    _ϕ               = 1.0 / ϕ

    # numerics
    # nz         = 63
    # ny         = nz
    # nx         = 2 * (nz + 1) - 1
    # nx, ny, nz = 255, 127, 127
    nx, ny, nz = nx, ny, nz  
    # nt         = 2000
    nt         = nt
    cfl        = 1.0 / sqrt(3.1)
    re_D       = 4π
    maxiter    = 10max(nx, ny, nz)
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx, ny, nz))
    # preprocessing
    dx, dy, dz           = lx / nx, ly / ny, lz / nz
    _dx, _dy, _dz        = 1.0 / dx, 1.0 / dy, 1.0 / dz 
    xn, yn, zn           = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly / 2, ly / 2, ny + 1), LinRange(-lz, 0, nz + 1)
    xc, yc, zc           = av1(xn), av1(yn), av1(zn)
    θ_dτ_D               = max(lx, ly, lz) / re_D / cfl / min(dx, dy, dz)
    _1_θ_dτ_D            = 1.0 / (1.0 + θ_dτ_D)
    β_dτ_D               = (re_D * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
    _β_dτ_D              = 1.0 / β_dτ_D
    # init
    Pf                  = @zeros(nx, ny, nz)
    r_Pf                = @zeros(nx, ny, nz)
    qDx, qDy, qDz       = @zeros(nx + 1, ny, nz), @zeros(nx, ny + 1, nz), @zeros(nx, ny, nz+1)
    qDx_c, qDy_c, qDy_c = zeros(nx, ny, nz), zeros(nx, ny, nz) , zeros(nx, ny, nz)
    qDmag               = zeros(nx, ny, nz)
    T    = [ΔT * exp(-xc[ix]^2 - yc[iy]^2 - (zc[iz] + lz / 2)^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz]
    T[:, :, 1  ] .= ΔT / 2
    T[:, :, end] .= -ΔT / 2
    T    = Data.Array(T)
    T_old = copy(T)
    dTdt                = @zeros(nx - 2, ny - 2, nz - 2)
    r_T                 = @zeros(nx - 2, ny - 2, nz - 2)
    qTx                 = @zeros(nx - 1, ny - 2, nz - 2)
    qTy                 = @zeros(nx - 2, ny - 1, nz - 2)
    qTz                 = @zeros(nx - 2, ny - 2, nz - 1)
    # action
    # vis
    if do_viz
        ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end
        loadpath = "viz3D_out/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)")
        iframe = 0
    end
    # time loop
    for it = 1:nt
        T_old .= T
        # time step
        dt = if it == 1
            0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy)), dz / maximum(abs.(qDz))) / 3.1)
        end
        _dt = 1.0 / dt
        re_T        = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T      = max(lx, ly, lz) / re_T / cfl / min(dx, dy, dz)
        _1_θ_dτ_T   = 1.0 / (1.0 + θ_dτ_T)
        β_dτ_T      = (re_T * λ_ρCp) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
        _dt_β_dτ_T  = 1.0 / (1.0 / dt + β_dτ_T)
        # iteration loop
        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # hydro
            @parallel compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρgx, αρgy, αρgz, T)
            @parallel update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
            # thermo
            @parallel compute_Tflux!(qTx, qTy, qTz, qDx, qDy, qDz, dTdt, λ_ρCp, T, T_old, _dx, _dy, _dz, _dt,  _1_θ_dτ_T, _ϕ)
            @parallel update_T!(T, dTdt, qTx, _dx, qTy, _dy, qTz, _dz, _dt_β_dτ_T)
            @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
            @parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)
            if (iter % ncheck == 0) && do_check
                @parallel compute_r!(r_Pf, r_T, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz)
                err_D = maximum(abs.(r_Pf))
                err_T = maximum(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx, err_D, err_T)
            end
            iter += 1
        end
        @printf("it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
        # visualisation

        if do_viz && (it % nvis == 0)
            ENV["GKSwstype"] = "nul"
            if (isdir("viz3D_out") == false) mkdir("viz3D_out") end
            loadpath = "viz3D_out/"
            p1 = heatmap(xc, zc, Array(T)[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
            png(p1, @sprintf("viz3D_out/%04d.png", iframe += 1))
            save_array("out_T", convert.(Float32, Array(T)))
            save_array("out_Pf", convert.(Float32, Array(Pf)))
        end
    end
    return T
end

if test==false
    porous_convection_3D(do_viz=true, do_check=true)
else
end

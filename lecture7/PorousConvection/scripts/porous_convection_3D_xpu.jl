const USE_GPU = false
using ParallelStencil
# using ParallelStencil.FiniteDifferences2D
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end
using Printf, Plots

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

@parallel function compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρgx, αρgy, αρgz, T)
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) * _dx - αρgx * @av_xa(T))) * _1_θ_dτ_D
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) * _dy - αρgy * @av_ya(T))) * _1_θ_dτ_D
    @inn_z(qDz) = @inn_z(qDz) - (@inn_z(qDz) + k_ηf * (@d_za(Pf) * _dz - αρgz * @av_za(T))) * _1_θ_dτ_D
    return nothing
end

@parallel function update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz) * _β_dτ_D 
    return nothing
end

@parallel function compute_Tflux!(qTx, qTy,qTz,  λ_ρCp, T, _dx, _dy, _dz, _1_θ_dτ_T)
    @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * @d_xi(T) * _dx ) * _1_θ_dτ_T
    @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * @d_yi(T) * _dy ) * _1_θ_dτ_T
    @all(qTz) = @all(qTz) - (@all(qTz) + λ_ρCp * @d_zi(T) * _dz ) * _1_θ_dτ_T
    return nothing 
end

@parallel function update_T!(T, dTdt, qTx, _dx, qTy, _dy, qTz, _dz, _dt_β_dτ_T)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz) * _dt_β_dτ_T
    return nothing
end

@parallel_indices (iy, iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end


@views function porous_convection_3D(; nx = 255, ny = 127, nz = 127, nt = 2000, do_viz = false)
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
    xn, yn, zn           = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly, 0, ny + 1), LinRange(0, lz, nz + 1)
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
    T                   = Data.Array([ΔT * exp(-xc[ix]^2 - yc[iy]^2 - (zc[iz] + lz / 2)^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz])
    T[:, 1,:]          .= ΔT / 2
    T[:, end,:]        .= -ΔT / 2
    T_old               = copy(T)
    dTdt                = @zeros(nx - 2, ny - 2, nz - 2)
    r_T                 = @zeros(nx - 2, ny - 2, nz - 2)
    qTx                 = @zeros(nx - 1, ny - 2, nz - 2)
    qTy                 = @zeros(nx - 2, ny - 1, nz - 2)
    qTz                 = @zeros(nx - 2, ny - 2, nz - 1)
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
            @parallel compute_Tflux!(qTx, qTy,qTz,  λ_ρCp, T, _dx, _dy, _dz, _1_θ_dτ_T)
            dTdt .= (T[2:end-1, 2:end-1, 2:end-1] .- T_old[2:end-1, 2:end-1, 2:end-1]) .* _dt .+
                    (max.(qDx[2:end-2, 2:end-1, 2:end-1], 0.0) .* diff(T[1:end-1, 2:end-1, 2:end-1]; dims=1) .* _dx .+
                     min.(qDx[3:end-1, 2:end-1, 2:end-1], 0.0) .* diff(T[2:end  , 2:end-1, 2:end-1]; dims=1) .* _dx .+
                     max.(qDy[2:end-1, 2:end-2, 2:end-1], 0.0) .* diff(T[2:end-1, 1:end-1, 2:end-1]; dims=2) .* _dy .+
                     min.(qDy[2:end-1, 3:end-1, 2:end-1], 0.0) .* diff(T[2:end-1, 2:end  , 2:end-1]; dims=2) .* _dy .+
                     max.(qDz[2:end-1, 2:end-1, 2:end-2], 0.0) .* diff(T[2:end-1, 2:end-1, 1:end-1]; dims=3) .* _dz .+
                     min.(qDz[2:end-1, 2:end-1, 3:end-1], 0.0) .* diff(T[2:end-1, 2:end-1, 2:end  ]; dims=3) .* _dz
                     ) .* _ϕ
            @parallel update_T!(T, dTdt, qTx, _dx, qTy, _dy, qTz, _dz, _dt_β_dτ_T)

            @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)

            if iter % ncheck == 0
                r_Pf  .= diff(qDx; dims=1) ./ dx .+ diff(qDy; dims=2) ./ dy .+ diff(qDz; dims=3) ./ dz
                r_T   .= dTdt .+ diff(qTx; dims=1) ./ dx .+ diff(qTy; dims=2) ./ dy .+ diff(qTz; dims=3) ./ dz
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
            if (isdir("viz_out3d") == false) mkdir("viz_out3d") end
            loadpath = "viz_out3d/"
            anim = Animation(loadpath, String[])
            p1 = heatmap(xc, zc, Array(T)[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
            png(p1, @sprintf("viz_out3d/%04d.png", iframe += 1))
        end

    end
    save_array("out_T", convert.(Float32, Array(T)))
    return T
end

# porous_convection_3D(do_viz=true)
# porous_convection_3D(nx = 30, ny = 30, nz = 10, nt = 50)

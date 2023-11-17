using Printf, LazyArrays, Plots

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

@views function porous_convection_2D()
    # physics
    lx, ly     = 40.0, 20.0
    k_ηf       = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2)
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    nx      = 127#200
    ny      = ceil(Int, nx * ly / lx)
    nt      = 500
    re_D    = 4π
    cfl     = 1.0 / sqrt(2.1)
    maxiter = 10max(nx, ny)
    ϵtol    = 1e-6
    nvis    = 20
    ncheck  = ceil(0.25max(nx, ny))
    # preprocessing
    dx, dy = lx / nx, ly / ny
    xn, yn = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly, 0, ny + 1)
    xc, yc = av1(xn), av1(yn)
    θ_dτ_D = max(lx, ly) / re_D / cfl / min(dx, dy)
    β_dτ_D = (re_D * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    # init
    Pf           = zeros(nx, ny)
    r_Pf         = zeros(nx, ny)
    qDx, qDy     = zeros(nx + 1, ny), zeros(nx, ny + 1)
    qDx_c, qDy_c = zeros(nx, ny), zeros(nx, ny)
    qDmag        = zeros(nx, ny)
    T            = @. ΔT * exp(-xc^2 - (yc' + ly / 2)^2)
    T[:, 1]      .= ΔT / 2
    T[:, end]    .= -ΔT / 2
    T_old        = copy(T)
    dTdt         = zeros(nx - 2, ny - 2)
    r_T          = zeros(nx - 2, ny - 2)
    qTx          = zeros(nx - 1, ny - 2)
    qTy          = zeros(nx - 2, ny - 1)
    # vis
    st     = ceil(Int, nx / 25)
    Xc, Yc = [x for x in xc, y in yc], [y for x in xc, y in yc]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    # action
    for it in 1:nt
        T_old .= T
        # time step
        dt     = if it == 1
            0.1 * min(dx, dy) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1)
        end
        re_T   = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T = max(lx, ly) / re_T / cfl / min(dx, dy)
        β_dτ_T = (re_T * λ_ρCp) / (cfl * min(dx, dy) * max(lx, ly))
        # iteration loop
        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # hydro
            qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (Diff(Pf; dims=1) ./ dx .- αρgx .* avx(T))) ./ (1.0 + θ_dτ_D)
            qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (Diff(Pf; dims=2) ./ dy .- αρgy .* avy(T))) ./ (1.0 + θ_dτ_D)
            Pf              .-= (Diff(qDx; dims=1) ./ dx .+ Diff(qDy; dims=2) ./ dy) ./ β_dτ_D
            # thermo
            qTx .-= (qTx .+ λ_ρCp .* (Diff(T[:, 2:end-1]; dims=1) ./ dx)) ./ (1.0 + θ_dτ_T)
            qTy .-= (qTy .+ λ_ρCp .* (Diff(T[2:end-1, :]; dims=2) ./ dy)) ./ (1.0 + θ_dτ_T)
            dTdt .= (T[2:end-1, 2:end-1] .- T_old[2:end-1, 2:end-1]) ./ dt .+
                    (max.(qDx[2:end-2, 2:end-1], 0.0) .* Diff(T[1:end-1, 2:end-1]; dims=1) ./ dx .+
                     min.(qDx[3:end-1, 2:end-1], 0.0) .* Diff(T[2:end  , 2:end-1]; dims=1) ./ dx .+
                     max.(qDy[2:end-1, 2:end-2], 0.0) .* Diff(T[2:end-1, 1:end-1]; dims=2) ./ dy .+
                     min.(qDy[2:end-1, 3:end-1], 0.0) .* Diff(T[2:end-1, 2:end  ]; dims=2) ./ dy) ./ ϕ
            T[2:end-1, 2:end-1] .-= (dTdt .+ Diff(qTx; dims=1) ./ dx .+ Diff(qTy; dims=2) ./ dy) ./ (1.0 / dt + β_dτ_T)
            T[[1, end], :] .= T[[2, end - 1], :]
            if iter % ncheck == 0
                r_Pf  .= Diff(qDx; dims=1) ./ dx .+ Diff(qDy; dims=2) ./ dy
                r_T   .= dTdt .+ Diff(qTx; dims=1) ./ dx .+ Diff(qTy; dims=2) ./ dy
                err_D = maximum(abs.(r_Pf))
                err_T = maximum(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx, err_D, err_T)
            end
            iter += 1
        end
        @printf("it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
        # visualisation
        if it % nvis == 0
            qDx_c .= avx(qDx)
            qDy_c .= avy(qDy)
            qDmag .= sqrt.(qDx_c .^ 2 .+ qDy_c .^ 2)
            qDx_c ./= qDmag
            qDy_c ./= qDmag
            qDx_p = qDx_c[1:st:end, 1:st:end]
            qDy_p = qDy_c[1:st:end, 1:st:end]
            heatmap(xc, yc, T'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo)
            display(quiver!(Xp[:], Yp[:]; quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black))
        end
    end
    return
end

porous_convection_2D()

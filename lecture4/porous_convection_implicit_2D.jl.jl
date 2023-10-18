using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function porous_convection_2D()
    # physics
    lx      = 40.0
    ly      = 20.0
    k_ηf    = 1.0
    αρgx,αρgy = 0.0,1.0
    αρg       = sqrt(αρgx^2+αρgy^2)
    ΔT        = 200.0
    ϕ         = 0.1
    Ra        = 1000.0
    λ_ρCp     = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    nvis      = 5
    # numerics
    
    nx      = 127
    ny      = ceil(Int, nx*ly/lx)
    ϵtol    = 1e-8
    maxiter = 100max(nx,ny)
    ncheck  = ceil(Int, 0.25max(nx,ny))
    cfl     = 1.0/sqrt(2.1)
    re_D    = 4π

    # derived numerics
    dx      = lx / nx
    dy      = ly / ny
    dt_diff   = min(dx,dy)^2/λ_ρCp/4.1
    xc      = LinRange(-lx/2+dx/2, lx/2+dx/2, nx)
    yc      = LinRange(-ly +dy/2, -dy/2, ny)
    θ_dτ_D    = max(lx,ly)/re_D/cfl/min(dx,dy)
    β_dτ_D    = (re_D*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))

    # array initialisation
    Pf = zeros(Float64,nx,ny)
    #Pf       = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2)
    r_Pf = zeros(Float64, nx, ny)
    T         = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2)
    T_old     = zeros(Float64, size(T))
    T[:,1] .= ΔT/2
    T[:,end] .= -ΔT/2
    dTdt        = zeros(nx-2,ny-2)
    r_T         = zeros(nx-2,ny-2)
    qTx         = zeros(nx-1,ny-2)
    qTy         = zeros(nx-2,ny-1)
    #T[[1,end],:] .= T[[2,end-1],:]
    dTdt_adv = zeros(Float64, nx-2, ny-2)
    qTx      = zeros(Float64, nx + 1, ny)
    qTy      = zeros(Float64, nx, ny - 1)
    qDx      = zeros(Float64, nx + 1, ny)
    qDy      = zeros(Float64, nx, ny + 1)

    # visualisation init
    st        = ceil(Int,nx/25)
    Xc, Yc    = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp    = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]
    #visualisation
    qDxc     = zeros(Float64, nx, ny)
    qDyc     = zeros(Float64, nx, ny)
    qDmag    = zeros(Float64, nx, ny)
    #time loop
    nt = 500
    for it = 1:nt
        T_old .= T
        # time st  
        dt =1
        dt = if it == 1
        re_T    = π + sqrt(π^2 + ly^2/λ_ρCp/dt)
        θ_dτ_T  = max(lx,ly)/re_T/cfl/min(dx,dy)
        β_dτ_T  = (re_T*λ_ρCp)/(cfl*min(dx,dy)*max(lx,ly))
            0.1*min(dx,dy)/(αρg*ΔT*k_ηf)
        else
            min(5.0*min(dx,dy)/(αρg*ΔT*k_ηf),ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1)
        end

        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol 
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # fluid pressure update
            qDx[2:end-1,:] .-= 1.0./(1.0 + θ_dτ_D) .* (qDx[2:end-1, :] .+ k_ηf .* (diff(Pf, dims=1) ./ dx .- αρgx .* 0.5 .* (T[1:end-1, :] .+ T[2:end, :])))
            qDy[:,2:end-1] .-= 1.0./(1.0 + θ_dτ_D) .* (qDy[:, 2:end-1] .+ k_ηf .* (diff(Pf, dims=2) ./ dy .- αρgy .* 0.5 .* (T[:, 1:end-1] .+ T[:, 2:end])))
            Pf              .-=  (diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy )./ β_dτ
            # temperature updat
            qTx[2:end-1,:] .= .- λ_ρCp .* diff(T, dims=1) ./ dx
            qTy            .= .- λ_ρCp .* diff(T, dims=2) ./ dy
            dTdt           .= (T[2:end-1,2:end-1] .- T_old[2:end-1,2:end-1])./dt .+ (qDx[2:end-1,:].+qDy[:,2:end-1]).* (diff(T, dims=1)./dx .+ diff(T,dims=2)./dy)./ϕ
            dt_adv = ϕ * min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
            dt     = min(dt_diff, dt_adv)
            T[2:end-1,2:end-1] .-= (dTdt .+ qDx .* diff(T, dims=1)./dx .+ qDy .* diff(T, dims=2)./dy)./(1.0/dt + β_dτ_T)
            T[[1,end],:]       .= T[[2,end-1],:]
            
            if iter % ncheck == 0
                r_Pf  .= diff(qDx, dims=1)./dx +diff(qDy, dims=2)./dy
                err_D = maximum(abs.(r_Pf))
                push!(iter_evo, iter / nx); push!(err_evo, err_D)  
                @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n",it,iter/nx,err_D)  
            end
            iter += 1

            if iter % ncheck == 0
                r_Pf  .= diff(qDx, dims=1)./dx +diff(qDy, dims=2)./dy
                r_T   .= dTdt .+ diff(qTx, dims=1)./dx +diff(qTy, dims=2)./dy
                err_D  = maximum(abs.(r_Pf))
                err_T  = maximum(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",iter/nx,err_D,err_T)
            end

             # Temperature update explicit
            dt_adv = ϕ * min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
            dt     = min(dt_diff, dt_adv)
            qTx[2:end-1,:] .= .- λ_ρCp .* diff(T, dims=1) ./ dx
            qTy            .= .- λ_ρCp .* diff(T, dims=2) ./ dy
            T[:, 2:end-1] .-= dt .* (diff(qTx[:, 2:end-1], dims=1) ./ dx .+ diff(qTy, dims=2) ./ dy)
            dTdt_adv .= 0.0
            dTdt_adv .-= max.(qDx[2:end-2,2:end-1] ./ ϕ, 0.0) .* diff(T[1:end-1, 2:end-1], dims=1) ./ dx
            dTdt_adv .-= min.(qDx[3:end-1,2:end-1] ./ ϕ, 0.0) .* diff(T[2:end  , 2:end-1], dims=1) ./ dx
            dTdt_adv .-= max.(qDy[2:end-1,2:end-2] ./ ϕ, 0.0) .* diff(T[2:end-1, 1:end-1], dims=2) ./ dy
            dTdt_adv .-= min.(qDy[2:end-1,3:end-1] ./ ϕ, 0.0) .* diff(T[2:end-1, 2:end  ], dims=2) ./ dy
            T[2:end-1, 2:end-1] .+= dt .* dTdt_adv
            end

        #=temperature evolution
        dt_adv = ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
        dt     = min(dt_diff,dt_adv)
        qTx[2:end-1,:] .= diff(T,dims=1)./dx
        qTy .= diff(T,dims=2)./dy

        T[2:end-1,2:end-1] .+= dt.*λ_ρCp.*(diff(qTx[2:end-1,2:end-1], dims =1) ./ dx .+diff(qTy[2:end-1,:], dims=2) ./ dy)
        T[2:end-1,2:end-1] .-= dt./ϕ.*(diff(qDx[2:end-1,2:end-1], dims =1)./dx .+diff(qDy[2:end-1,2:end-1], dims=2)./dy)
        =#
       


        # visualisation
        if it % nvis == 0
            qDxc  .= 0.5 .* (qDx[1:end-1, :] .+ qDx[2:end, :])
            qDyc  .= 0.5 .* (qDy[:, 1:end-1] .+ qDy[:, 2:end])
            qDmag .= sqrt.(qDxc.^2 .+ qDyc.^2)
            qDxc  ./= qDmag
            qDyc  ./= qDmag
            qDx_p = qDxc[1:st:end, 1:st:end]
            qDy_p = qDyc[1:st:end, 1:st:end]
            heatmap(xc,yc,T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo)
            display(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black))
        end
        
    end

end

porous_convection_2D()
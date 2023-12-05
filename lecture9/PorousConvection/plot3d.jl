using GLMakie, Printf

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise()
    for iframe in 1:20
        lx, ly, lz = 40.0, 20.0, 20.0
        nx = 506
        ny = 250
        nz = 125
        T  = zeros(Float32, nx, ny, nz)
        println(@sprintf("viz3Dmpi_out/out_T_%04d", iframe))
        load_array(@sprintf("viz3Dmpi_out/out_T_%04d", iframe), T)
        xc, yc, zc = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)
        fig = Figure(resolution=(1600, 1000), fontsize=24)
        ax  = Axis3(fig[1, 1]; aspect=(1, 1, 0.5), title="Temperature", xlabel="lx", ylabel="ly", zlabel="lz")
        surf_T = contour!(ax, xc, yc, zc, T; alpha=0.05, colormap=:turbo)
        save(@sprintf("viz3Dmpi_out/out_T_%04d.png", iframe), fig)
    end
    return 
end

visualise()
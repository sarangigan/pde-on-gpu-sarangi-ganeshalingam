using MAT, Plots

file = matopen("docs/multi_xpu_2D_out_C_0_0651.mat")
C_0 = read(file, "C_v"); close(file)

file = matopen("docs/gpu_2D_out_C_0_0651.mat")
C_gpu = read(file, "C"); close(file)

println(size(C_0), size(C_gpu[2:end-1,2:end-1]))
println(sum(C_0.^2 .- C_gpu[2:end-1,2:end-1].^2))
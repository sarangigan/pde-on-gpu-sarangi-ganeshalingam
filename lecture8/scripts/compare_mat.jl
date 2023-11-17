using MAT

file = matopen("docs/cpu_2D_out_C_0_0651.mat")
C_cpu = read(file, "C"); close(file)
file = matopen("docs/gpu_2D_out_C_0_0651.mat")
C_gpu = read(file, "C"); close(file)

println(sum(C_cpu.^2 .- C_gpu.^2))
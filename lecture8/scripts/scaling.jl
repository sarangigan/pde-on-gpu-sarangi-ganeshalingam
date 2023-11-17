using Plots, Plots.Measures, Printf, DelimitedFiles, LaTeXStrings
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=2, labelfontsize=11, tickfontsize=11, titlefontsize=11)


# Strong scaling
#Read files

T_eff = readdlm("docs/strong_scaling.txt")

length_ls = 16 * 2 .^ (1:10)

#plotting
plot(length_ls, T_eff,xlabel="nx",ylabel=L"T_{eff}",label="Strong Scaling",marker=:circle)

hline!([559], linestyle=:dash,label="Nvidia Tesla P100 GPUs")
savefig("./docs/strong_scaling.png")

# Weak scaling
# Time =  0.675384 sec for nx = ny = 16 * 2 ^10 for single GPU
# 0.704903 4 gpu
# 0.715465 16gpu
# 0.732396 25 gpu
# 0.737644 64gpu

t_ls = [0.675384, 0.704903, 0.715465, 0.732396, 0.737644]
t_ls_norm = t_ls./t_ls[1]
plot([1,4,16,25,64], t_ls_norm, xlabel="number of gpu node", ylabel=L"t/t_1", label="Weak Scaling",marker=:circle)
savefig("./docs/weak_scaling.png")

# @hide_communication
# 0.748813 (0,0)
# 0.738977 (2,2)
# 0.737644 (8,2)
# 0.754603 (16,4)
# 0.767212 (16,16)

t_ls = [0.748813, 0.738977, 0.737644, 0.754603, 0.767212]
t_ls_norm = t_ls ./ 0.675384
x_ls = 1:size(t_ls,1)
plot(x_ls, t_ls_norm, xlabel="@hide_communication" ,ylabel="normalized computation time", label="hide_communication",marker=:circle)
xticks!(x_ls, ["no comm", "(2,2)", "(8,2)", "(16,4)", "(16,16)"])
savefig("./docs/hide_communication.png")




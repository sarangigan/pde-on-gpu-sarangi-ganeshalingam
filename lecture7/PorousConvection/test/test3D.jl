using PorousConvection
using Test

include("../scripts/porous_convection_3D_xpu.jl")

T = porous_convection_3D(nx = 10, ny = 10, nz = 10, nt = 10)


# a unit test
@testset "PorousConvection.jl" begin
    # Write your tests here.
    @test @views porous_convection_3D(nx = 10, ny = 10, nz = 10, nt = 10) == porous_convection_3D(nx = 10, ny = 10, nz = 10, nt = 10)
end


# a reference test:
# make inds with `sort(rand(1:length(X), 10)
xinds = sort(rand(1:size(T,1), 30))
yinds = sort(rand(1:size(T,2), 30))
zinds = sort(rand(1:size(T,3), 10))

# make vals with `X[inds]`
vals_T  = T[xinds, yinds, zinds]

@test all(vals_T  .≈ T[xinds, yinds, zinds])
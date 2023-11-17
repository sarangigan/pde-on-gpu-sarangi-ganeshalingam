using PorousConvection
using Test

include("../scripts/porous_convection_xpu.jl")

T = porous_convection_2D(nx = 10, ny = 10, nt = 10)


# a unit test
@testset "PorousConvection.jl" begin
    # Write your tests here.
    @test @views porous_convection_2D(nx = 10, ny = 10, nt = 10) == porous_convection_2D(nx = 10, ny = 10, nt = 10)
end


# a reference test:
# make inds with `sort(rand(1:length(X), 10)
xinds = sort(rand(1:size(T,1), 30))
yinds = sort(rand(1:size(T,2), 30))

# make vals with `X[inds]`
vals_T  = T[xinds, yinds]


@test all(vals_T  .≈ T[xinds, yinds])

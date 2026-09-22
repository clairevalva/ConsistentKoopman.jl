using Distances
using LinearAlgebra
function make_distances_2(X)
    R = pairwise(euclidean, X, dims=2)
    return R
end

# test that these are the same, and then also if they work
include("../examples/torusData.jl") 
x_test = x[1:2, 1:10]
_, nT = size(x_test)
D = make_distances_2(x_test)

i = 8
j = 2

x = x_test[:, i]
xpre = x_test[:, i - 1]
y = x_test[:, j]
ypre = x_test[:, j - 1]

vx = x .- xpre
vy = y .- ypre

vx = vx / norm(vx, 2)
vy = vy / norm(vy, 2)

diffxy = x .- y
normdiff = norm(diffxy, 2)
diffxy = diffxy / norm(diffxy, 2)

cθ1 = -vx' * diffxy
cθ2 = vy' * diffxy

ζ = 0.995
sqrt((1 - ζ*cθ1) * (1 - ζ*cθ2))
ans = conebw(x, y, xpre, ypre)

include("../src/kernelsMatrix.jl")
vxmat, c1mat, c2mat, diffs, testBWsmat = coneBandwidths(x_test, D, returnall = true)
vxmat[:, i - 1] == vx
vxmat[:, j - 1] == vy

D[i - 1, j - 1] - normdiff
norm(diffxy, 2) == 1
(norm(diffs[:, i - 1, j - 1], 2) - 1) .< 1e-8
(diffs[:, i - 1, j - 1] - diffxy) .< 1e-8

cθ1 - c1mat[i - 1, j - 1] .< 1e-8
cθ2 - c2mat[i - 1, j - 1] .< 1e-8

testBWsmat[i - 1, j - 1] - ans^-1 .< 1e-8


# sep bandwith test

test, _ = est_ind_bandwidth(D, nT, nT)
testold = sepBandwidths(D, 1)

test .- testold
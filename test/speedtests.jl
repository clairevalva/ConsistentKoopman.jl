using Distances
using BenchmarkTools
using LinearAlgebra

# make some data
include("../examples/torusData.jl") 

# test distance function speed
function make_distances_1(X, nT)
    D = zeros(Float64, nT, nT)
    # if no nearest neighbors specified, keep all of them
    for i = 1:nT
        for j = 1:(i-1)
            D[j,i] = euclidean(X[:,j], X[:,i])
        end
    end
    D = D + D'
    return D
end

function make_distances_2(X)
    R = pairwise(euclidean, X, dims=2)
    return R
end

# test that these are the same, and then also if they work
x_test = x[:, 1:500]
_, nT = size(x_test)
D1 = make_distances_1(x_test, nT)
D2 = make_distances_2(x_test)
mean(D1 - D2) <= 1e-16

time1 = @benchmark make_distances_1(x_test, nT)
# 3.571 ms (249504 allocations: 26.66 MiB)

time2 = @benchmark make_distances_2(x_test)
# 3.571 ms (249504 allocations: 26.66 MiB)

using ForwardDiff
include("../src/kernelsMatrix.jl")
candidate_ϵs = 2 .^ (range(-40,40,length = 100))

S = make_Sϵ(D2, 1)
S(1)
bw, m = tuneBandwidth(D2, 1, candidate_ϵs)

S_grad(γ) = ForwardDiff.derivative(S, γ)
sgrads = S_grad.(candidate_ϵs)

eps_ls = candidate_ϵs
eps_sum  = S.(candidate_ϵs)

eps_sum_prime = (eps_sum[2:end] - eps_sum[1:(end - 1)]) ./ (eps_ls[2:end] - eps_ls[1:(end - 1)])
argmax(eps_sum_prime)
eps_sum_prime[45]
2^eps_ls[45]

# answers seem close enough I think for me


vx = test2 .- circshift(test2, 1)

norm.(eachrow(vx), 2)

# test cone bws!
testBWs = coneBandwidths(x_test, D2)

i = 104
j = 17
conebw(x_test[:, i], x_test[:, j], x_test[:, i - 1], x_test[:, j - 1] )

testBWs[j, i]

testBWs


# is checking worse that something else


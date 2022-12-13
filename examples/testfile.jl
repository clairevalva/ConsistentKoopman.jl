# TODO test current
# TODO projection to physical space (and reconstruction?)
include("torusData.jl")
include("kernels.jl")
include("modelConstruct.jl")
include("modelComponents.jl")
using Plots
using Statistics



# testμ, testη, testφ = lapEig(X, 1., 0, 51)
# TODO : test when NN != 0

# plot(real(testφ[1:1000,2]))
# plot!(imag(testφ[1:1000,3]))

# write down params
X = Matrix(x');
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nT = size(X, 1) - 1
symM = false
nDiff = 50

# compute distances, reset NN if necessary
D, DN = distNN(X, NN)

if NN == 0
    NN_bw = nT
else
    NN_bw = NN
end


# tune bandwith params
useϵ, testϵ_ls = tune_bandwidth(D, DN, NN_bw, nT, candidate_ϵs)

# optimal guess was epsilonOpt = 0.1408 in original NLSA 
# plot(candidate_ϵs[1:end - 1], testϵ_ls, xaxis=:log )
# vline!([0.1408])

# compute sparse kernel matrix
W = sparseW_mb(X, useϵ, NN = 0)

# normalize matrix
P, μ = normW(W)

# compute diffusion eigenfunctions
κ, φ = computeDiffusionEig(P, nDiff)

# # normalize eigenfunctions as in from 10.1016/j.acha.2017.09.001
# normφ = zeros(size(φ))
# for k = 1:nDiff
#     normφ[:,k] = φ[:,k] ./ norm(φ[:,k])
# end
# normφ = normφ ./ mean(φ[:,1])
# η = log.(κ) ./ log.(κ[1])

normφ, η = normDiffEigs(φ)

plot(dt*(1:1000), real(normφ[1:1000,2]))
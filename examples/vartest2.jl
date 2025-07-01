# test to see if IOPs made from different methods are the same or not

include("../src/ConsistentKoopman.jl")
include("../src/physinformed/kernels.jl")
include("../src/physinformed/kerneltuning.jl")

using LinearAlgebra
using Distances
using BenchmarkTools
using LinearOperators
using LowRankApprox

# get and reduce data
include("torusData.jl")
X =  x[:, 1:500]
usedt = dt
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = minimum(size(X, 2), 100)
N = size(X, 2)

NLSAmodel = ConsistentKoopman.paramsNLSA(X, usedt, NN, candidate_ϵs, nDiff)
NLSAresults, m̂, bw = ConsistentKoopman.doNLSAMatrix(NLSAmodel, "seperable", m = NaN)

φ = NLSAresults.φ # NLSA eigenfunctions
κ = NLSAresults.κ # diffusion coefficients (NLSA eigenvalues)
w = NLSAresults.w # weights for ℓ² inner product, = 1 / nT

ι(g) = inclusion(g, X)

# do variational with same params
k_sb = makesepbwk(X, m̂*2, euclidean)
k_use(x, y) =  k_sb(x, y, bw)
k_bs = bssym_norm(k_use, X)

K_bs = makeIO(k_bs, X)

function K_bs_inc(v)
    return ι(K_bs(v))
end

K_bs(X[:, 2])

K_bs_inc(φ[:, 7])
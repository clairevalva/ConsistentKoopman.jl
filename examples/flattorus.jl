include("torusData.jl")
include("../src/kernels.jl")
include("../src/utils.jl")
include("../src/modelComponents.jl")
include("../src/plot_utils.jl")
include("../src/resolvent.jl")
include("../src/modelstructs.jl")
using Plots
using Statistics
using StatsBase

# NLSA
X =  Matrix(x');
usedt = dt
NN = 4000;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 201

NLSAmodel = paramsNLSA(X, usedt, NN, candidate_ϵs, nDiff)
NLSAresults = doNLSA(NLSAmodel)

# Koopman
nKoop = 101
mKoop = ceil(Int64, nKoop / 3)
z = 1.0
τ = 1e-4
koopmodel = makeParamsKoop(NLSAresults, nKoop, mKoop, z, τ )
koopresults = doKoopman(koopmodel)

ω, ζ = koopresults.ω, koopresults.ζ
max_eps, sortinds = sortautocorr(ζ, ω, 1000, usedt, returnall = false)


# nT = size(X, 1) - 1
# if NN == 0
#     NN_bw = nT
# else
#     NN_bw = NN
# end

# # compute NLSA 
# D, DN = distNN(X, NN)
# useϵ, m̂ = tune_bandwidth(D, DN, NN_bw, nT, candidate_ϵs)
# # W = sparseW_mb(X, useϵ, usedt, NN = NN, sym = true)
# W = sparseW_sepband(X, useϵ, m̂, D, DN, NN = NN, sym = true)
# P = normW(W)
# κ, φ, w = computeDiffusionEig(P, nDiff)

# # compute Koopman 
# φ_plus = posfilter(φ)
# G = Gtau(κ, τ)[1:nKoop, 1:nKoop]
# Rz = resolventop_power(φ_plus, w, 50, usedt, z)[1:nKoop, 1:nKoop]

# ω, ζ, c = computeSeigs(Rz, G, z, nKoop, mKoop, φ)
# max_eps, sortinds = sortautocorr(ζ, ω, 1000, usedt, returnall = false)

include("../src/ConsistentKoopman.jl")
using SparseArrays
using LinearAlgebra
include("torusData.jl")

# NLSA
X =  Matrix(x');
usedt = dt
NN = 4000;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 201
NLSAmodel = ConsistentKoopman.paramsNLSA(X, usedt, NN, candidate_ϵs, nDiff)

X = NLSAmodel.srcdata
NN = NLSAmodel.NN
candidate_ϵs = NLSAmodel.candidate_ϵs
nDiff = NLSAmodel.nDiff

nT = size(X, 1)

if NN == 0
    NN_bw = nT
else
    NN_bw = NN
end

print("computing distances")
D, DN = ConsistentKoopman.distNN(X, NN)
print("computing bandwidth")
useϵ, m̂ = ConsistentKoopman.tune_bandwidth(D, DN, NN_bw, nT, candidate_ϵs)
print("sparseW")
W = ConsistentKoopman.sparseW_sepband(X, useϵ, m̂, D, DN, NN = NN, sym = true)
P = ConsistentKoopman.normW(W)
κ, φ, w = ConsistentKoopman.computeDiffusionEig(P, nDiff)
eigsNLSA(params, κ, φ, w)

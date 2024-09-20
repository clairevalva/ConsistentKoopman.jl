include("../src/ConsistentKoopman.jl")
using SparseArrays
using ForwardDiff
using LinearAlgebra
include("torusData.jl")
include("/kontiki6/cnv5172/koopmanInteraction/temp_nlsa_files.jl")

# NLSA
X =  Matrix(x);
usedt = dt
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 201
nKoop = 201
mKoop = Int(ceil(nKoop / 6))
τ = 1e-4
z = 1.
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
σ2 = make_σ2_sepbw(D, DN)
# σ2 = make_σ2_conebw(X, D)

# function k(xy, σ2; ϵ)
#     return exp(-1*xy^2 / (σ2 * ϵ^2))
# end

function k(xy, σ2; ϵ)
        return exp(-1*(xy * σ2)^2 / ϵ^2)
end

println("computing bandwidth")
candidate_ϵs = (range(-40,40,length = 100))
bestϵ, estdim = est_bw(D[2:end, 2:end], σ2, k, candidate_ϵs)
# bestϵ, estdim = est_bw(D, σ2, k, candidate_ϵs)
# useϵ = 0.1407997982348491
# m̂ = 2.156782950161558

# useϵ, m̂ = ConsistentKoopman.tune_bandwidth(D, DN, NN_bw, nT, candidate_ϵs)
print("sparseW")
# W = ConsistentKoopman.sparseW_sepband(X, useϵ, m̂, D, DN, NN = NN, sym = true)
σ2_m = σ2 #.^ (1  / estdim)
K = k.(D[2:end, 2:end], σ2_m, ϵ = bestϵ)
K[.~DN[2:end, 2:end]] .= 0
W = sparse(K)
W = check_sym(W)
W[(W .< 1e-16) .& (W .> 0)] .= 0.0
dropzeros!(W)


print("norm")
P = ConsistentKoopman.normW(W)
# P_sym = P * P'
κ, φ, w = ConsistentKoopman.computeDiffusionEig(P, nDiff)

print("koop")
φ_plus = ConsistentKoopman.posfilter(φ)
G = ConsistentKoopman.Gtau(κ, τ)[1:nKoop, 1:nKoop]
Rz = ConsistentKoopman.resolventop_power(φ_plus, w, 50, dt, z)[1:nKoop, 1:nKoop]
ω, ζ, c = ConsistentKoopman.computeSeigs(Rz, G, z, nKoop, mKoop, φ)

scatter(ω[ω .> 0], ylims = (0, 15))
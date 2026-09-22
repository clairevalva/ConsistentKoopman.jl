include("../examples/torusData.jl")
using LinearAlgebra
using SparseArrays
using Distances
using Statistics
include("../src/OOSestimation.jl")
include("../src/ConsistentKoopman.jl")

# NLSA step (RKHS basis)
X =  x[:, :]
usedt = dt
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 501
nT = size(X, 2)

println("computing distances")
D, _ = ConsistentKoopman.distNN(X, NN) # because NN = 0
# need to symmetrize DN

NN_bw = nT

σ = ConsistentKoopman.coneBandwidths(X, D)
# X = X[:, :] # X[:, 2:end]
D = D[2:end, 2:end]
nT -= 1

println("computing bandwidth (ϵ)")
bw, _ = ConsistentKoopman.tuneBandwidth(D, σ, candidate_ϵs)

println("make kernel matrix")
W = ConsistentKoopman.makeW(D, bw, σ)



println("normW")
# P_single = ConsistentKoopman.normW(W)
P_single = ConsistentKoopman.normW(W)
# P = normW_quad(W)
P = Symmetric(P_single)

println("compute Eigs")
κ, φ, w = ConsistentKoopman.computeDiffusionEig(P,  nDiff)


println("compute kernel evals")
# test everything works ok 
spotmatch = 48
testy = X[:,spotmatch  + 1]
testypre = X[:, spotmatch ]

testspot = 37

# check that it works
println("normalized kernel func")
p_faster = makeNormKernel_cone(W, X, nT, bw)
p_faster_single = makeNormKernel_cone_single(W, X, nT, bw)

p_evals, k_evals, khat_evals, dval, Qneghalf = p_faster(testy, testypre, verbose = true)

p_evals_single, _, _, _, _ = p_faster_single(testy, testypre, verbose = true)

# insert the k_evals from the original function?
k_evals_orig = W[:, spotmatch]
maximum(abs.(k_evals_orig - k_evals))
dval_2 = sum(k_evals_orig)

khat_evals_orig = k_evals_orig * (dval_2 ^-1) 
Qneghalf, K̂ = normWPieces_single(W)
k_sum_orig = zeros(nT)
for l = 1:nT
    jvals = zeros(nT)
    for j = 1:nT
        # k_sum[l] += khat_evals[j] *  K̂[j, l] * Qneghalf[j] # replace this naive summing alg with something better
        jvals[j] = khat_evals_orig[j] *  K̂[j, l] * Qneghalf[j]
    end
    k_sum_orig[l] = sum(jvals)
end


println("check that kernel func matches normalized matrix: ", P[spotmatch, testspot] - p_evals[testspot])

println("check that kernel func matches normalized matrix (single): ", P_single[spotmatch, testspot] - p_evals_single[testspot])


println("check that kernel func matches normalized matrix (double only kernel evals): ", P_single[spotmatch, testspot] - k_sum_orig[testspot])

# now attempt to do a direct phi compuation
testk = 27
# phi_val =  (nT * κ[testk])^(-1) * sum(p_evals .* φ[:, testk])

correct_sum = φ[spotmatch, testk]

phi_calc = sum(p_evals_single .* φ[:, testk]) * (κ[testk]^(-1))

mat_calc = sum(P[:, spotmatch] .* φ[:, testk]) * (κ[testk]^(-1))

phi_rel_diff = Float64((correct_sum - phi_calc) / correct_sum)
matrix_rel_diff = Float64((correct_sum - mat_calc) / correct_sum)

println("what is the relative difference (matrix calc)? ",matrix_rel_diff)
println("what is the relative difference (phi calc)? ", phi_rel_diff)
# try with just a gaussian RBF or a NN?
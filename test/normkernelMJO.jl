using LinearAlgebra
using SparseArrays
using Distances
using Statistics
using HDF5

include("../src/OOSestimation.jl")
include("../src/ConsistentKoopman.jl")

# NLSA step (RKHS basis)
X_load = h5read("test/merid_avgs_predict_6_2025.h5", "olr_u")
Y = reshape(X_load, :, 9132)
Y = Y[:, 1:6000]
usedt = 1
nEmb = 1
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 501
nT = size(Y, 2) - (nEmb - 1)

println("computing distances")
X = ConsistentKoopman.delayembed(Y, nEmb)
# X = Y

# D, DN = ConsistentKoopman.distNN(X, NN, nEmb)
D, DN = ConsistentKoopman.distNN(X, NN)
NN_bw = nT

DN_sym = symDist(DN)
NN_bw = nT
println("symmetry, number entries changed: ", sum(DN_sym .!== DN))
println("number entries changed: ", sum(DN_sym .!== DN))

# delayE = ConsistentKoopman.delayembed(X, nEmb)
# test that these match
# testj = 57
# testi = 500

# euclidean(delayE[:, testj], delayE[:, testi])
# D[testi, testj]

# σ = ConsistentKoopman.coneBandwidths(delayE, D)
σ = ConsistentKoopman.coneBandwidths(X, D)

D = D[2:end, 2:end]
DN_sym = DN_sym[2:end, 2:end]
nT -= 1


println("computing bandwidth (ϵ)")
bw, _ = ConsistentKoopman.tuneBandwidth(D, σ, candidate_ϵs)

println("make kernel matrix")
W = ConsistentKoopman.makeW(D, bw, σ)
W_NN = copy(W)
W_NN[.~DN_sym] .= 0

println("NN, number entries changed: ", sum(W_NN .!== W))
W = W_NN

println("normW")
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
p_faster_single_NN = makeNormKernel_cone_single(W, X, nT, bw, NN)

p_evals, k_evals, khat_evals, dval, Qneghalf = p_faster(testy, testypre, verbose = true)

p_evals_single, _, _, _, _ = p_faster_single(testy, testypre, verbose = true)
p_evals_single_NN, _, _, _, _ = p_faster_single_NN(testy, testypre, verbose = true)

println("check that kernel func matches normalized matrix: ", P[spotmatch, testspot] - p_evals[testspot])
println("check that kernel func matches normalized matrix (NN): ", P[spotmatch, testspot] - p_evals_single_NN[testspot])

println("check that kernel func matches normalized matrix (single): ", P_single[spotmatch, testspot] - p_evals_single[testspot])


# now attempt to do a direct phi compuation
testk = 27
# phi_val =  (nT * κ[testk])^(-1) * sum(p_evals .* φ[:, testk])

correct_sum = φ[spotmatch, testk]

phi_calc = sum(p_evals_single .* φ[:, testk]) * (κ[testk]^(-1))
phi_calc_NN = sum(p_evals_single_NN .* φ[:, testk]) * (κ[testk]^(-1))
mat_calc = sum(P[:, spotmatch] .* φ[:, testk]) * (κ[testk]^(-1))

phi_rel_diff = Float64((correct_sum - phi_calc) / correct_sum)
phi_rel_diff_NN = Float64((correct_sum - phi_calc_NN) / correct_sum)
matrix_rel_diff = Float64((correct_sum - mat_calc) / correct_sum)

println("what is the relative difference (matrix calc)? ",matrix_rel_diff)
println("what is the relative difference (phi calc)? ", phi_rel_diff)
println("what is the relative difference (phi calc, NN)? ", phi_rel_diff_NN)

# try with just a gaussian RBF or a NN?
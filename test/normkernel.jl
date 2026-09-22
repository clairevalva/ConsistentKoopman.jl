include("../examples/torusData.jl")
using LinearAlgebra
using SparseArrays
using Distances
using Statistics
include("../src/OOSestimation.jl")
include("../src/ConsistentKoopman.jl")

# NLSA step (RKHS basis)
X =  x[:, 1:1000]
usedt = dt
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 501
nT = size(X, 2)

println("computing distances")
D, DN = ConsistentKoopman.distNN(X, NN)
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
P = ConsistentKoopman.normW(W)
P = normW_quad(W)
P = Symmetric(P)

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

W2 = Double64.(W)
Dsum = sum(W2, dims = 1)[:]
Dinv = Diagonal(Dsum.^(-1))
Q = sum(W2 ./ (Dsum'), dims = 2)[:]
Qneghalf2 = Diagonal(Q.^(-1/2))

K̂ = Dinv * W2 * Qneghalf2

Dsum_n2 = sum(W, dims = 1)[:]
Dinv_n2 = Diagonal(Dsum_n2.^(-1))
Q_n2 = sum(W ./ (Dsum_n2'), dims = 2)[:]
Qneghalf2_n2 = Diagonal(Q_n2.^(-1/2))

K̂_n2 = Dinv_n2 * W * Qneghalf2_n2

println("quad precision d mat diff: ", maximum(abs.(Dsum_n2 -Dsum  )))
println("quad precision d inv mat diff: ", maximum(abs.(Dinv_n2 -Dinv  )))
println("quad precision khat mat diff: ", maximum(abs.(K̂_n2 -K̂  )))

Q2 = diag(Qneghalf2)

p_evals, k_evals, khat_evals, dval, Qneghalf = p_faster(testy, testypre, verbose = true)

p_evals_single, _, _, _, _ = p_faster_single(testy, testypre, verbose = true)

onevec =ones(Double64, size(k_evals))

k = makebwKer(bw)



println("max error W: ", maximum(abs.(W2[:, spotmatch] .- k_evals)))
println("max error D: ", abs.(dval - Dsum[spotmatch]))

println("max error khat eval: ", maximum(abs.(khat_evals - K̂[:, spotmatch])))



println("check that kernel func matches normalized matrix: ", P[spotmatch, testspot] - p_evals[testspot])

println("check that kernel func matches normalized matrix (single): ", P[spotmatch, testspot] - p_evals_single[testspot])

# now attempt to do a direct phi compuation
testk = 8
# phi_val =  (nT * κ[testk])^(-1) * sum(p_evals .* φ[:, testk])

correct_sum = φ[spotmatch, testk]

phi_calc = sum(p_evals .* φ[:, testk]) * (κ[testk]^(-1))
mat_calc = sum(P[:, spotmatch] .* φ[:, testk]) * (κ[testk]^(-1))

phi_rel_diff = Float64((correct_sum - phi_calc) / correct_sum)
matrix_rel_diff = Float64((correct_sum - mat_calc) / correct_sum)

println("what is the relative difference (matrix calc)? ",matrix_rel_diff)
println("what is the relative difference (phi calc)? ", phi_rel_diff)

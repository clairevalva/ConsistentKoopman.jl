using LinearAlgebra
using Distances
using Statistics
using Dates
using BenchmarkTools
using HDF5
# include("../../src/OSE.jl")
include("../../src/ConsistentKoopman.jl")

# do a test with RMM data, expect the NN error to be less than in torus case
X_load = h5read("/scratch/cnv5172/kontiki6/koopmanInteraction/compatdata/merid_avgs_predict_6_2025.h5", "olr_u")
X = reshape(X_load, :, size(X_load, 3))

expname = "init"
tStart = 1
tEnd = 7305
NN = 1500;
nEmb = 64 # nEmb = 1 and 0 are equivalent to match notation in other papers



println("start: ", Date(2000, 01, 01) + Day(tStart - 1), ", end: ", Date(2000, 01, 01) + Day(tEnd - 1))

savename = "/scratch/cnv5172/koopmanMJO/data/" * expname * "_NLSA.h5"

X = X[:, tStart:tEnd]


# set test indices
testk = 11 # phi recon place
spotmatch = 300 # kernel recon place

nT = size(X, 2)
if nEmb > 1
    Xemb = ConsistentKoopman.delayembed(X, nEmb)
    nT = size(Xemb, 2)
end

usedt = 1
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = min(501, nT - 1)


println("computing distances")
# D, DN = ConsistentKoopman.distNN(X, NN, nEmb) # because NN = 0

# need to symmetrize DN

# if NN > 0
#     DN_sym = ConsistentKoopman.symDist(DN)
#     println("symmetry, number entries changed: ", sum(DN_sym .!== DN))
#     DN = DN_sym
# end

D = h5read(savename, "D")
DN = h5read(savename, "DN")

# if nEmb > 1
#     σ = ConsistentKoopman.coneBandwidths(Xemb, D) # this could be relatively easily fixed to be fast
# else
#     σ = ConsistentKoopman.coneBandwidths(X, D)
# end
σ = h5read(savename, "sigma")


D = D[2:end, 2:end]
DN = DN[2:end, 2:end]
nT -= 1

# println("computing bandwidth (ϵ)")
# bw, _ = ConsistentKoopman.tuneBandwidth(D, σ, candidate_ϵs)

bw = h5read(savename, "bw")

println("make kernel matrix")
W = ConsistentKoopman.makeW(D, bw, σ)

if NN > 0
    W_NN = copy(W)
    W_NN[.~DN] .= 0
    cents = (nT - NN) * nT
    println("approximate number of entries changed: ", cents)
    println("number entries actually changed: ", sum(W_NN .!== W), "(", (sum(W_NN .!== W) - cents)/cents, ")")

    W = W_NN
end


println("normW")
P = ConsistentKoopman.normW(W)

# also get each matrix from this:
D_mat = sum(W, dims = 2)[:] 
Dinv = Diagonal(D_mat.^(-1))
S = sum(W ./ (D_mat'), dims = 2)[:]
Sneghalf = Diagonal(S.^(-1/2))
K̂ = Dinv * W * Sneghalf

println("compute Eigs")
# κ, φ, w = ConsistentKoopman.computeDiffusionEig(P, nDiff)

κ = h5read(savename, "kappa")
φ = h5read(savename, "phi")
w = size(φ, 2)


println("compute kernel evals")
# test everything works ok 

if nEmb > 1
    testy = Xemb[:, spotmatch  + 1]
    testypre = Xemb[:, spotmatch ]
else
    testy = X[:, spotmatch  + 1]
    testypre = X[:, spotmatch ]
end


# check that it works
println("normalized kernel func")
if nEmb > 1
    p_faster = ConsistentKoopman.makeNormKernel_cone(W, Xemb, nT, bw, NN)
else
    p_faster = ConsistentKoopman.makeNormKernel_cone(W, X, nT, bw, NN)
end
p_evals, k_evals, khat_evals, dval, Qneghalf, dist_evals = p_faster(testy, testypre, verbose = true)
k_sort = sortperm(dist_evals)
k_sort_D = sortperm(D[:, spotmatch])



# sorting difference:
println("max difference distance evals: ", maximum(abs.(dist_evals - D[:, spotmatch])))
println("max difference kernel eval: ", maximum(abs.(W[:, spotmatch] - k_evals)) )
println("number very different (NN sorting consequence)?: ", sum(abs.(W[:, spotmatch] - k_evals) .> 1e-8))

println("difference D: ", abs(D_mat[spotmatch] - dval))
println("max difference k̂: ", maximum(abs.(K̂[spotmatch, :] - khat_evals)))

println("check that kernel func matches normalized matrix: ", maximum(abs.(P[spotmatch, :] - p_evals)))



correct_sum = φ[spotmatch, testk]

phi_calc = sum(p_evals .* φ[:, testk]) * (κ[testk]^(-1))
mat_calc = sum(P[:, spotmatch] .* φ[:, testk]) * (κ[testk]^(-1))
phi_calc_test = ConsistentKoopman.evalPhi(p_evals, φ[:, testk], κ[testk])

println("function vs manually calculated: ", phi_calc - phi_calc_test)

phi_rel_diff = Float64((correct_sum - phi_calc) / correct_sum)
matrix_rel_diff = Float64((correct_sum - mat_calc) / correct_sum)

println("what is the relative difference (matrix calc)? ",matrix_rel_diff)
println("what is the relative difference (phi calc)? ", phi_rel_diff)
# # try with just a gaussian RBF or a NN?

println("fin")

# max difference distance evals: 4.850822883408227
# max difference kernel eval: 0.9329265564028016
# number very different (NN sorting consequence)?: 187
# difference D: 125.25223453362901
# max difference k̂: 0.0007147540849026286
# check that kernel func matches normalized matrix: 2.7242773173406695e-5
# function vs manually calculated: 0.0
# what is the relative difference (matrix calc)? 7.21983299329218e-15
# what is the relative difference (phi calc)? -0.009299715777642761
# fin

# do some benchmarking

@benchmark p_faster(testy, testypre, verbose = true)

# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 68.360 s (0.97% GC) to evaluate,
#  with a memory estimate of 28.66 GiB, over 254978360 allocations.

# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 49.352 s (1.54% GC) to evaluate,
#  with a memory estimate of 28.66 GiB, over 254978144 allocations.

# this sucks but is livable I think, given that this projects everything
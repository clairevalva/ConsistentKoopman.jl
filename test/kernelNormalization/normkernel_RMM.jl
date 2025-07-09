using LinearAlgebra
using Distances
using Statistics
using HDF5
include("../../src/OSE.jl")
include("../../src/ConsistentKoopman.jl")

# do a test with RMM data, expect the NN error to be less than in torus case
X_load = h5read("test/merid_avgs_predict_6_2025.h5", "olr_u")
X = reshape(X_load, :, 9132)
# X = X[:, 1:4000]
NN = 1500;
nEmb = 64 # nEmb = 1 and 0 are equivalent to match notation in other papers

# set test indices
testk = 17 # phi recon place
spotmatch = 3000 # kernel recon place

nT = size(X, 2)
if nEmb > 1
    Xemb = ConsistentKoopman.delayembed(X, nEmb)
    nT = size(Xemb, 2)
end

usedt = 1
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = min(501, nT - 1)


println("computing distances")
D, DN = ConsistentKoopman.distNN(X, NN, nEmb) # because NN = 0
# need to symmetrize DN

if NN > 0
    DN_sym = symDist(DN)
    println("symmetry, number entries changed: ", sum(DN_sym .!== DN))
    DN = DN_sym
end

if nEmb > 1
    σ = ConsistentKoopman.coneBandwidths(Xemb, D) # this could be relatively easily fixed to be fast
else
    σ = ConsistentKoopman.coneBandwidths(X, D)
end

D = D[2:end, 2:end]
DN = DN[2:end, 2:end]
nT -= 1

println("computing bandwidth (ϵ)")
bw, _ = ConsistentKoopman.tuneBandwidth(D, σ, candidate_ϵs)

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
κ, φ, w = ConsistentKoopman.computeDiffusionEig(P, nDiff)


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
    p_faster = makeNormKernel_cone(W, Xemb, nT, bw, NN)
else
    p_faster = makeNormKernel_cone(W, X, nT, bw, NN)
end
p_evals, k_evals, khat_evals, dval, Qneghalf, dist_evals = p_faster(testy, testypre, verbose = true)
k_sort = sortperm(dist_evals)
k_sort_D = sortperm(D[:, spotmatch])

# sorting difference:
println("max difference distance evals: ", maximum(abs.(dist_evals - D[:, spotmatch])))

println("max difference kernel eval: ", maximum(abs.(W[:, spotmatch] - k_evals)) )
println("number very different (NN sorting consequence)?: ", sum(abs.(W[:, spotmatch] - k_evals) .> 1e-8))


plot(W[:, spotmatch] - k_evals)

println("difference D: ", abs(D_mat[spotmatch] - dval))
println("max difference k̂: ", maximum(abs.(K̂[spotmatch, :] - khat_evals)))

println("check that kernel func matches normalized matrix: ", maximum(abs.(P[spotmatch, :] - p_evals)))



correct_sum = φ[spotmatch, testk]

phi_calc = sum(p_evals .* φ[:, testk]) * (κ[testk]^(-1))
mat_calc = sum(P[:, spotmatch] .* φ[:, testk]) * (κ[testk]^(-1))

phi_rel_diff = Float64((correct_sum - phi_calc) / correct_sum)
matrix_rel_diff = Float64((correct_sum - mat_calc) / correct_sum)

println("what is the relative difference (matrix calc)? ",matrix_rel_diff)
println("what is the relative difference (phi calc)? ", phi_rel_diff)
# # try with just a gaussian RBF or a NN?

println("fin")


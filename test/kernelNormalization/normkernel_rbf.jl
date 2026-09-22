include("../examples/torusData.jl")
using LinearAlgebra
using SparseArrays
using Distances
using Statistics
include("../src/OSE.jl")
include("../src/ConsistentKoopman.jl")

# this works with NN at good tolerances as far as I can tell

X =  x[:, :]
usedt = dt
NN = 500;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = min(501, size(X, 2))
nT = size(X, 2)

println("computing distances")
D, DN = ConsistentKoopman.distNN(X, NN) # because NN = 0
# need to symmetrize DN

if NN > 0
    DN_sym = symDist(DN)
    println("symmetry, number entries changed: ", sum(DN_sym .!== DN))
end

println("computing bandwidth (ϵ)")
bw, _ = ConsistentKoopman.tuneBandwidth(D, 1, candidate_ϵs)

println("make kernel matrix")
W = ConsistentKoopman.makeW(D, bw, 1)

if NN > 0
    W_NN = copy(W)
    W_NN[.~DN_sym] .= 0
    cents = (nT - NN) * nT
    println("approximate number of entries changed: ", cents)
    println("number entries actually changed: ", sum(W_NN .!== W), "(", (sum(W_NN .!== W) - cents)/cents, ")")

    W = W_NN
end


println("normW")
P = ConsistentKoopman.normW(W)

# also get each matrix from this:
D = sum(W, dims = 2)[:] 
Dinv = Diagonal(D.^(-1))
S = sum(W ./ (D'), dims = 2)[:]
Sneghalf = Diagonal(S.^(-1/2))
K̂ = Dinv * W * Sneghalf

println("compute Eigs")
κ, φ, w = ConsistentKoopman.computeDiffusionEig(P,  nDiff)


println("compute kernel evals")
# test everything works ok 
spotmatch = 2
testy = X[:,spotmatch]

testspot = 3

# check that it works
println("normalized kernel func")
p_faster = makeNormKernel_rbf(W, X, nT, bw, NN)
p_evals, k_evals, khat_evals, dval, Qneghalf = p_faster(testy, verbose = true)

println("max difference kernel eval: ", maximum(abs.(W[:, spotmatch] - k_evals)) )
println("difference D: ", abs(D[spotmatch] - dval))
println("max difference k̂: ", maximum(abs.(K̂[spotmatch, :] - khat_evals)))

println("check that kernel func matches normalized matrix: ", P[spotmatch, testspot] - p_evals[testspot])


# now attempt to do a direct phi compuation
testk = 7
correct_sum = φ[spotmatch, testk]

phi_calc = sum(p_evals .* φ[:, testk]) * (κ[testk]^(-1))
mat_calc = sum(P[:, spotmatch] .* φ[:, testk]) * (κ[testk]^(-1))

phi_rel_diff = Float64((correct_sum - phi_calc) / correct_sum)
matrix_rel_diff = Float64((correct_sum - mat_calc) / correct_sum)

println("what is the relative difference (matrix calc)? ",matrix_rel_diff)
println("what is the relative difference (phi calc)? ", phi_rel_diff)
# # try with just a gaussian RBF or a NN?

println("fin")
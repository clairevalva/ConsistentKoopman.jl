include("../examples/torusData.jl")
using LinearAlgebra
using SparseArrays
include("../src/ConsistentKoopman.jl")

# NLSA step (RKHS basis)
X =  x
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

println("normalized kernel func")
k_faster, k_norm = makeNormKernel(W, X, nT, bw)

# try it individual
Qneghalf, K̂ = normWPieces(W)

 # assemble initial kernel
function k(x, y, xpre, ypre)
    D = euclidean(x, y)
    σ = ConsistentKoopman.conebw(x, y, xpre, ypre).^-1
    return exp.(-1 * D ./ (σ .^ 2 * bw .^ 2))
end


k_test = 19
testy = X[:, 19]
testypre = X[:, 18]



k_evals = zeros(nT)
for j = 1:nT
    k_evals[j] += k(testy, X[:, j + 1], testypre, X[:, j])
end

dval = sum(k_evals) / nT
khat_evals = (k_evals ./ dval) .* Qneghalf

k_sum = zeros(nT)
for l = 1:nT
    for j = 1:nT
        k_sum[l] += khat_evals[j] *  K̂[j, l]
    end
end

k_sum = k_sum / nT
testk_evals = k_faster(testy, testypre)

testk_evals == k_sum


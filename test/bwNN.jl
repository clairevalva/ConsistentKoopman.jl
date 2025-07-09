include("../examples/torusData.jl")
using LinearAlgebra
using SparseArrays
using Distances
using Statistics
include("../src/OSE.jl")
include("../src/ConsistentKoopman.jl")


X = x[:, 1:200]
NN = 0;
nEmb = 2 # nEmb = 1 and 0 are equivalent to match notation in other papers

nT = size(X, 2)
if nEmb > 1
    Xemb = ConsistentKoopman.delayembed(X, nEmb)
    nT = size(Xemb, 2)
end

usedt = dt
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

i = 10
j = 12
nD = size(X, 1)

vx = Xemb[:, 2:nT] .- Xemb[:, 1:(nT - 1)]
vx = vx ./ norm.(eachcol(vx), 2)'


diffxy = (Xemb[:, i + 1] - Xemb[:, j + 1]) #/ D[i + 1, j + 1]
cθ1 = -1*vx[:,i]' * diffxy
cθ2  = vx[:,j]' * diffxy


# I should just be able to add these together before the square root
vx_f = X[:, 2:nT] .- X[:, 1:(nT - 1)]
vx_fnorm = norm.(eachcol(ConsistentKoopman.delayembed(vx_f, nEmb)), 2)
vx_f = vx_f[1:(nT - nEmb)] ./ vx_fnorm # divide this part at the end


Xemb[1:nD, i] == X[:, i + 1]
Xemb[(nD + 1):(2*nD), i] == X[:, i]

diffxy_1 = (X[:, i + 1] - X[:, j + 1]) 
diffxy_2 = (X[:, i ] - X[:, j ]) 
cθ1_1 = (-1*vx_f[:,i]' * diffxy_1 + -1*vx_f[:,i - 1]' * diffxy_2) #/ D[i + 1, j + 1]



# check that vectors work like I think
testvec = rand(8)
testvec2 = rand(8)

var1 = testvec' * testvec2

var2a = testvec[1:nD]' * testvec2[1:nD]
var2b = testvec[(nD + 1):(2*nD)]' * testvec2[(nD + 1):(2*nD)]
var2 = var2a + var2b

diff = var1 - var2
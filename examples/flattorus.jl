using ConsistentKoopman
include("torusData.jl")


# NLSA
X =  Matrix(x');
usedt = dt
NN = 4000;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 201

NLSAmodel = paramsNLSA(X, usedt, NN, candidate_ϵs, nDiff)
NLSAresults = doNLSA(NLSAmodel)

# Koopman
nKoop = 101
mKoop = ceil(Int64, nKoop / 3)
z = 1.0
τ = 1e-4
koopmodel = makeParamsKoop(NLSAresults, nKoop, mKoop, z, τ )
koopresults, Rz = doKoopman(koopmodel)

ω, ζ = koopresults.ω, koopresults.ζ
max_eps, sortinds = sortautocorr(ζ, ω, 1000, usedt, returnall = false)


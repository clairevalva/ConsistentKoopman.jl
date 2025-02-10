# using ConsistentKoopman
include("../src/ConsistentKoopman.jl")
include("torusData.jl")


# NLSA
X =  x
usedt = dt
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 501

NLSAmodel = ConsistentKoopman.paramsNLSA(X, usedt, NN, candidate_ϵs, nDiff)
NLSAresults = ConsistentKoopman.doNLSA(NLSAmodel, "cone")

φ = NLSAresults.φ
κ = NLSAresults.κ
w = NLSAresults.w

# test alt function
include("../src/fdGenerator.jl")
Vtest = makeVf(φ, dt)
Δ = makeΔ(κ)

λ, u, ν, gs =  doKoopman_diff(φ, κ, dt, 1E-9)
# i think this is working, just nor really getting high frequencies? worth more tests but need to work on kernel part first

# Koopman
nKoop = 101
mKoop = ceil(Int64, nKoop / 3)
z = 1.0
τ = 1e-4
koopmodel = ConsistentKoopman.makeParamsKoop(NLSAresults, nKoop, mKoop, z, τ )
koopresults, Rz = ConsistentKoopman.doKoopman(koopmodel)
φ_plus = ConsistentKoopman.posfilter(φ)
G = ConsistentKoopman.Gtau(κ, τ)[1:nKoop, 1:nKoop]
Rz = ConsistentKoopman.resolventop_power(φ_plus, w, 50, dt, z)[1:nKoop, 1:nKoop]
ω, ζ, c = ConsistentKoopman.computeSeigs(Rz, G, z, nKoop, mKoop, φ)

ω, ζ = koopresults.ω, koopresults.ζ
max_eps, sortinds = sortautocorr(ζ, ω, 1000, usedt, returnall = false)


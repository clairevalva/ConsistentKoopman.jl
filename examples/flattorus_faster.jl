# explores functionality of package in the flat torus example
include("../src/ConsistentKoopman.jl")
include("torusData.jl")
using LinearAlgebra

# NLSA step (RKHS basis)
X =  x
usedt = dt
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nDiff = 501

NLSAmodel = ConsistentKoopman.paramsNLSA(X, usedt, NN, candidate_ϵs, nDiff)
NLSAresults = ConsistentKoopman.doNLSAMatrix(NLSAmodel, "seperable", m = NaN)

φ = NLSAresults.φ # NLSA eigenfunctions
κ = NLSAresults.κ # diffusion coefficients (NLSA eigenvalues)
w = NLSAresults.w # weights for ℓ² inner product, = 1 / nT

# check φ are orthonormal
println("⟨φj, φj⟩ = ", sum(φ[:, 7] .* φ[:, 7]) * w[7])
println("⟨φj, φi⟩ = ", sum(φ[:, 7] .* φ[:, 19]) * w[7])

## Koopman method 1 (apply regularization directly to generator, i.e., https://doi.org/10.1016/j.acha.2021.02.004)
Vtest = ConsistentKoopman.makeVf(φ, dt)
Δ = ConsistentKoopman.makeΔ(κ)
ω_m1, u, _, ψ_m1 =  ConsistentKoopman.doKoopman_diff(φ, κ, dt, 1E-9)

# compare to perfect sine wave
j = 6
perfect = exp.(1im*imag(ω_m1[j])*(0:500)*dt)
p1 = plot((0:500)*dt, imag(perfect), label = "perfect")
plot!((0:500)*dt, imag(ψ_m1[1:501,j]), label = "approximate")
plot!(title = "Koopman method 1, ω = " * string(round(ω_m1[j], digits = 5)),
    xlabel = "time", ylabel = "imag(ψ)")

# compare sorting methods!
maxϵ_m1, sortϵ_m1 = ConsistentKoopman.sortautocorr(ψ_m1, imag(ω_m1), 100, usedt, returnall = false)
E_m1 = ConsistentKoopman.computeDirichletE(u, κ)
p_ind_m1 = scatter(1:50, sortϵ_m1[1:50], ylabel = "autocorr sort", xlabel = "dirichlet energy sort", aspectratio = :equal, label = "")
p_ind_m2 = scatter(E_m1[1:50], maxϵ_m1[1:50], ylabel = "autocorr error", xlabel = "dirichlet energy", label = "")
plot(p_ind_m1, p_ind_m2, layout = (1,2), figsize = (300, 850), suptitle = "sorting method comparison")


## Koopman method 2 (apply regularization to the resolvent, i.e., https://dx.doi.org/10.1088/1361-6544/ad4ade)
nKoop = 201
mKoop = ceil(Int64, nKoop / 3)
z = 1.0
τ = 1e-4
koopmodel = ConsistentKoopman.makeParamsKoop(NLSAresults, nKoop, mKoop, z, τ )
koopresults, Rz = ConsistentKoopman.doKoopman(koopmodel)
ω_m2, ψ_m2, c = koopresults.ω, koopresults.ζ, koopresults.c
c = c[:, 1:mKoop]
ω_m2 = ω_m2[1:mKoop]
ψ_m2 = ψ_m2[:, 1:mKoop]

# sort autocorr and compute Dirichlet energy
maxϵ_m2, sortϵ_m2 = ConsistentKoopman.sortautocorr(ψ_m2, ω_m2, 100, usedt, returnall = false)
# E_m2 = sum( abs.( c) .^ 2 ./ κ, dims = 1)[:]
E_m2 = ConsistentKoopman.computeDirichletE(c, κ)

ω_m2 = ω_m2[sortϵ_m2]
ψ_m2 = ψ_m2[:, sortϵ_m2]
c = c[:, sortϵ_m2]

# compare to perfect sine wave
j = 2
perfect = exp.(1im*ω_m2[j]*(0:300)*dt)
p2 = plot((0:300)*dt, imag(perfect), label = "perfect")
plot!((0:300)*dt, imag(ψ_m2[1:301,j]), label = "approximate")
plot!(title = "Koopman method 2, ω = " * string(round(ω_m2[j], digits = 5)),
    xlabel = "time", ylabel = "imag(ψ)")




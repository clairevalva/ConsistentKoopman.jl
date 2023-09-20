# TODO test current
# TODO projection to physical space (and reconstruction?)
include("torusData.jl")
include("../src/kernels.jl")
include("../src/utils.jl")
include("../src/modelComponents.jl")
include("../src/plot_utils.jl")
include("../src/resolvent.jl")
using Plots
using Statistics
using HDF5
using StatsBase

# write down params
X =  Matrix(x');
usedt = dt
NN = 8000;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nT = size(X, 1) - 1
symM = false
nDiff = 201
nKoop = 201
z = 1.0
# compute distances, reset NN if necessary
D, DN = distNN(X, NN)

if NN == 0
    NN_bw = nT
else
    NN_bw = NN
end

# tune bandwith params, check if it worked
useϵ, testϵ_ls = tune_bandwidth(D, DN, NN_bw, nT, candidate_ϵs)

plot(candidate_ϵs[1:end - 1], testϵ_ls, xaxis=:log )
vline!([0.1408, useϵ]) # optimal guess was epsilonOpt = 0.1408 in original NLSA 

W = sparseW_mb_2(X, useϵ, usedt, NN = NN, sym = false)
P = normW2(W) # normalize matrix appropriately
κ, φ, w = computeDiffusionEig(P, nDiff) # compute (nomalized) diffusion eigenfunctions


φ_plus = posfilter(φ_old)
plot((1:4000)*dt, real(φ_plus[1:4000,8]))
plot!((1:4000)*dt, imag(φ_plus[1:4000,8]))
xlims!(0,5)

φ_plus = posfilter(φ)
plot((1:4000)*dt, real(φ_plus[1:4000,6]))
plot!((1:4000)*dt, imag(φ_plus[1:4000,6]))
xlims!(0,5)

### ALL KOOPMAN STUFF TESTED


G = Gtau(κ, 5e-5)
# G = Gtau(κ_old, 5e-5)
Rz = resolventop_power(φ_plus, w, 50, usedt, z)

Rz = Rz[1:nKoop, 1:nKoop]
G = G[1:nKoop, 1:nKoop]
U, P = polar(Rz)
Ptau = G * P * G
F = eigen(Ptau)
c = F.vectors
γ_polar = real(F.values)

gamma = polarfz(γ_polar, z)
frequencies = imag(1 ./gamma)
frequencies[isnan.(frequencies)] .= 0
ζ = makeζ(c, φ[:,1:nKoop])
max_eps, sortinds, acs, goalacs = sortautocorr(ζ, frequencies, 1000, usedt, returnall = true)
ζ_sorted = ζ[:, sortinds]
freqs_sorted = frequencies[sortinds]
ts = collect((1:500)*dt)
fs = ζ_sorted[1:500, :]
fsac = acs[1:500, sortinds]
fsgoal = goalacs[1:500, sortinds]
Nplot = 20
subtitles = ["f = " * string(f) for f in freqs_sorted ]
pbig = plotfuns(fs, ts, Nplot, pltimag = true, subtitles = subtitles)
plot!(size = (900, 100*Nplot))
pac = plotfuns(fsgoal, ts, Nplot, pltimag = true, subtitles = subtitles)
plot!(size = (800, 100*Nplot))

testdiff = 1 .- abs.(fs .* fsgoal)

pac = plotfuns(testdiff, ts, Nplot, pltimag = true, subtitles = subtitles)
plot!(size = (800, 100*Nplot))
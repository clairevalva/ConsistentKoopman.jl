# TODO test current
# TODO projection to physical space (and reconstruction?)
include("torusData.jl")
include("../src/kernels.jl")
include("../src/utils.jl")
include("../src/modelComponents.jl")
include("../src/resolvent.jl")
using Plots
using Statistics
using HDF5

# write down params
X = Matrix(x');
NN = 0;
candidate_ϵs = 2 .^ (range(-40,40,length = 100))
nT = size(X, 1) - 1
symM = false
nDiff = 50
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
vline!([0.1408]) # optimal guess was epsilonOpt = 0.1408 in original NLSA 

W = sparseW_mb(X, useϵ, NN = 0) # compute sparse kernel matrix
P = normW2(W) # normalize matrix appropriately
κ, φ, w = computeDiffusionEig(P, nDiff) # compute (nomalized) diffusion eigenfunctions

# now try for the Koopman stuff?
G = Gtau(κ, 5e-5)
Rz = resolventop_power(φ, w, 50, dt, z)

U, P = polar(Rz)
Ptau = G * P * G
F = eigen(Ptau)
c = F.vectors
γ_polar = real(F.values)

gamma = polarfz(γ_polar, z)
frequencies = imag(1 ./gamma)

ζ = makeζ(c, φ)
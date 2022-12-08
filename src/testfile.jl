# TODO test current
# TODO projection to physical space (and reconstruction?)
include("torusData.jl")
include("origpaper.jl")
include("kernels.jl")
include("modelConstruct.jl")
using Plots

X = Matrix(x');
testμ, testη, testφ = lapEig(X, 1., 0, 51)

plot(real(testφ[1:1000,2]))
plot!(imag(testφ[1:1000,3]))

# optimal guess was epsilonOpt = 0.1408 in original NLSA 
NN = 0
D, DN = distNN(X, 0)
nT = size(X, 1) - 1
NN = nT
epsls = 2 .^ (range(-10,40,length = 60))
testeps, testlist = tune_bandwidth(D, DN, NN, nT, epsls)

plot(epsls[1:end - 1], testlist,xaxis=:log )
vline!([0.1408])
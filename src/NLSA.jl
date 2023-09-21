module NLSA

using LinearAlgebra
using Statistics
using FFTW
using StatsBase
using Plots

include("kernels.jl")
include("utils.jl")
include("modelComponents.jl")
include("resolvent.jl")
include("domodel.jl")
include("plotutils.jl")

end

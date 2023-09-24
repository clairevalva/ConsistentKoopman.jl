module ConsistentKoopman

using LinearAlgebra
using Statistics
using FFTW
using StatsBase
using Plots

include("kernels.jl")
include("utils.jl")
include("modelcomponents.jl")
include("resolvent.jl")
include("domodel.jl")
include("plotutils.jl")

end

module ConsistentKoopman

using LinearAlgebra
using Statistics
using FFTW
using StatsBase
using Plots
using SparseArrays
using Distances

include("kernels.jl")
include("utils.jl")
include("modelcomponents.jl")
include("resolvent.jl")
include("domodel.jl")
include("plotutils.jl")

end

include("../src/physinformed/kernels.jl")
include("../src/physinformed/kerneltuning.jl")

using Distances
using BenchmarkTools
using LinearOperators
using LowRankApprox

### build data / observations
N = 128
nD = 1
θs = collect(range(0, 2*pi - 2*pi/N, length = N))
function circ_to_R2(θ) # data prep and observation map 
    return [cos(θ), sin(θ)]
end

ι(g) = inclusion(g, θs)

θs_obs = zeros(2, N)
for j = 1:N
    θs_obs[:, j] = circ_to_R2(θs[j])
end

# random testing vectors
x1, x2 = θs_obs[:, 13], θs_obs[:, 57]
testV = rand(N)

### make and tune sep bandwidth kernel
# first make and tune in rbf space
function k_rbf(x,y, ϵ = 1)
    return exp(-1*euclidean(x, y)^2 / ϵ)
end


function dθdt(θ)
    return 1
end

# first tuning round
testϵs = -6:0.2:10.0
jϵ, ϵ, m̂ = tunek(θs_obs, k_rbf, testϵs)

# second round
k_sb = makesepbwk(θs_obs, m̂*2, euclidean)
jϵ, ϵ, m̂ = tunek(θs_obs, k_sb, testϵs)

k_use(x, y) =  k_sb(x, y, ϵ)
k_bs = bssym_norm(k_use, θs_obs)

k_sqr = sqrtsym_norm(k_use, θs_obs) # could be used for multiplication

K_bs = makeIO(k_bs, θs_obs)
K_bs_inc(v) = ι(K_bs(v))
K! = makekfun(K_bs_inc)

K_bs_lop = LinearOperators.LinearOperator(ComplexF64, N, N, true, true, K!)

# define derivatives for first and second arguments
v(t) = [-sin(t), cos(t)]
vk = make_vk(k_bs, v)

# ok now take eigendecomposition to make this works
# PROBABLY should check that kernel functions are the same
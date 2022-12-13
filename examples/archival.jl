## stuff to compute distances here
# pass kernel as k(x,y) = kspec(x,y, a, b)

function computeKdists(kernf::Function, X; symmetric = true)
    # assumes X is size nT by nD where nD is the spatial dimension
    nT = size(X,1)
    kDists = zeros(Float64, nT, nT)
    if symmetric
        # only do top half computations if need to
        for m = 1:nT
            for n = (m + 1):nT
                kDists[m,n] = kernf(X[m,:], X[n,:])
            end
        end
        kDists = kDists + kDists'
    else
        for m = 1:nT
            for n = 1:nT
                if n ==! m
                    kDists[m,n] = kernf(X[m,:], X[n,:])
                end
            end
        end
    end
    kDists = kDists + diagm(ones(Float64, nT))
    return kDists
end

function computeKdists_vel(kernf::Function, X; symmetric = true)
    # assumes X is size nT by nD where nD is the spatial dimension
    nT = size(X,1)
    kDists = zeros(Float64, nT, nT)
    if symmetric
        # only do top half computations if need to
        for m = 1:nT
            for n = (m + 1):nT
                kDists[m,n] = kernf(X[m,:], X[n,:], X[m - 1,:], X[n - 1,:])
            end
        end
        kDists = kDists + kDists'
    else
        for m = 1:nT
            for n = 1:nT
                if n ==! m
                    kDists[m,n] = kernf(X[m,:], X[n,:], X[m - 1,:], X[n - 1,:])
                end
            end
        end
    end
    kDists = kDists + diagm(ones(Float64, nT))
    return kDists
end

mutable struct NLSAmodel
    srcData::Array{Float64,2} # data should be size nT × nD 
    dt::Float64 # time resolution of data
    nDelays::Int # number of delays for embedding

    kernel::Function # kernel function of use
    kernelvel::Bool # does the kernel need more info? (i.e. is it computing velocities?)
    kernelargs::Array # extra arguments to the kernel
    nφ::Int
end

function doNLSA(model::NLSAmodel)
    # assumes that the kernel is symmetric here
    X = model.srcData
    Xemb = delayembed(X', model.nDelays)' # this takes array in the wrong shape
    
    if model.kernelvel
        ker(x,y, xmin, ymin) = model.kernel(x, y, xmin, ymin, model.kernelargs...)
        Kdists = computeKdists_vel(ker, Xemb)
    else
        ker(x,y) = model.kernel(x, y, model.kernelargs...)
        Kdists = computeKdists(ker, Xemb)
    end

    K = constructK(Kdists)
    η, φ = computeDiffusionEig(K, model.nφ)  

    return Xemb, η, φ
end



# original implementation, from supplement of 10.1073/pnas.1118984109
# input : data array x of size m × S
# lag window q
# Gaussian width ϵ
# number of nearest neighbors b
# number of Laplacian eigenfunctions l
# output: array of spatial modes u in embedding space, of size n × l, where n = mq
# array of temporal modes v of size s × l, where s = S − 2q + 1
# vector of singular values σ of size l
# arrays of spatio-temporal patterns {x˜
# 1
# , . . . , x˜
# l}, each of size m × s
# % m: physical space dimension
# % n: embedding space dimension
# % S: number of input samples
# % s: number of samples for which temporal modes are computed
# % because of embedding and the normalization by ξ in Algorithm S2, s < S
# % specifically, v(i, :) and ˜x
# k
# (:, i) correspond to x(:, i + 2q − 1)



include("kernels.jl")

function local_vel(X::Matrix{Float64})
    # input X, assumes X size nT by nD where nD is the spatial dimension
    nT = size(X, 1)
    vels = zeros(Float64, nT - 1)
    for j = 2:nT
        vels[j - 1] = norm(X[j,:] - X[j - 1, :])
    end
    return vels
end


function sparseW_vel(X::Matrix{Float64},eps, NN::Integer = 0; usenorm::Function = norm, tune = false)
    # only compute the distances for the nearest neighbors stuff
    D, N = distNN(X[1:(end - 1),:], NN, usenorm = usenorm)
    Vs = local_vel(X)
    
    nT = size(X, 1) - 1
    if NN == 0
        NN = nT
    end

    W = zeros(Float64, nT, nT)
    
    for i = 1:nT
        for j = 1:NN
            # in theory should be able to replace the below with simplevel_kernel
            divC = Vs[i] * Vs[N[i,j]] * eps
            W[i, N[i,j]] = exp(-1*D[i,j]^2 / divC )
        end
    end

    W = sym_M(W)
    return W
end



function lapEig(X::Matrix{Float64}, eps::Float64, NN = 0, L = 0; usenorm::Function = norm)
    W = sparseW_vel(X, eps, NN; usenorm = usenorm)
    P, μ = normW(W)
    η, φ = computeDiffusionEig(P, L)

    return μ, η, φ
end

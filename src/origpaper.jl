# original implementation, from supplement of 10.1073/pnas.1118984109
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

function distNN(X::Matrix{Float64}, NN::Integer = 0; usenorm::Function = norm)
    nT, _ = size(X)
    D = zeros(Float64, nT, nT)
    N = zeros(Integer, nT, nT)

    if NN == 0
        # if no nearest neighbors specified, keep all of them
        NN = nT
    end

    for i = 1:nT
        d = zeros(Float64, nT)
        for j = 1:nT
            d[j] = usenorm(X[i,:] .- X[j,:])
        end
        inds = sortperm(d)[1:NN]
        D[i,1:NN], N[i,1:NN] = d[inds], inds
    end
    # then have that D(i,j), d(N(j))
    return D, N
end

function sym_M(M::Matrix{Float64})
    N = size(M,1)
    # TO DO
    for i = 1:N
        for j = 2:i
            if (M[i,j] == 0) & (M[j,i] != 0)
            elseif (M[i,j] != 0) & (M[j,i] == 0)
                M[j,i] = M[i,j]
            end
        end
    end

    return M
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

function normW(X::Matrix{Float64})
    # at least runs with this sample matrix:
    # testX = 1.0*[1 2 ; 3 4]
    # assumes matrix is symmetric
    nX = size(X, 1)
    Q = sum(X, dims = 2)
    Qdiv = Q * Q'
    X = X ./ Qdiv

    Q = sum(X, dims = 1)
    for i = 1:nX
        X[i,:] = X[i,:] ./ Q[i]
    end
    # µ is a vector of size s with the property µP = µ
    μ = Q ./ sum(Q)

    return X, μ
end

function lapEig(X::Matrix{Float64}, eps::Float64, NN = 0, L = 0; usenorm::Function = norm)
    W = sparseW_vel(X, eps, NN; usenorm = usenorm)
    P, μ = normW(W)
    η, φ = computeDiffusionEig(P, L)

    return μ, η, φ
end

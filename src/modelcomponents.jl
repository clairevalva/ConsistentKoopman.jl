export 
    tune_bandwidth,
    sparseW_sepband,
    normW,
    computeDiffusionEig,
    diffSVD,
    diffProjection,
    projectDiffEig

"""
    tune_bandwidth(D::Matrix{Float64},
     DN::Matrix{Integer}, NN::Integer,
     nT::Integer, epsls::Vector{Float64})

    tune bandwidth function -> should make sure this works, the behavior is a lil funky
        fixed, is a small typo in the pseudo code I think (is missing a square?)

    from 10.1016/j.acha.2017.09.001
    probably cheating a bit by only taking the sum of the NN nearest neighbors
    hoping it doesn't matter too much 

    Arguments
    =================
    - D: precomputed distances for nearest neighbors NN (from distNN)
    - DN: distance indexing information (also from distNN)
    - NN: number of nearest neighbors
    - nT: total time of processed data
    - epsls: vector of test epsilons (strictly positive only)

"""
function tune_bandwidth(D::Matrix{Float64}, DN::Matrix{Integer}, 
    NN::Integer, nT::Integer, epsls::Vector{Float64})
    
    Dsq = D.^2
    ris, _ = est_ind_bandwidth(D, NN, nT) #sum(Dsq, dims = 2) ./ (NN - 1)

    nE = length(epsls)
    eps_sum = zeros(Float64, nE)
    
    for eee in 1:nE
        eps = epsls[eee]
        for m = 1:nT
            for n = 1:NN
                k = DN[m,n]
                eps_sum[eee] = eps_sum[eee] + exp(-1*Dsq[m,n] / (eps * ris[m] * ris[k])^2)
                
            end
        end
    end
    eps_sum = eps_sum / (nT^2)
    eps_sum_l = log.(eps_sum)
    epsls_l = log.(epsls)

    eps_sum_prime = (eps_sum_l[2:end] - eps_sum_l[1:(end - 1)]) ./ (epsls_l[2:end] - epsls_l[1:(end - 1)])
    best_eps = epsls[argmax(eps_sum_prime)]

    return best_eps, maximum(eps_sum_prime)
end


"""
    sparseW_sepband(X::Matrix{Float64}, eps::Float64, m̂::Float64, 
    D::Matrix{Float64}, N::Matrix{Integer}; NN::Integer = 0, sym::Bool = true)

    computes sparse kernel matrix W with gaussian multiple bandwidth kernel from given ϵ, see 10.1038/s41467-021-26357-x supplement and references therein

    Arguments
    =================
    - X: original data (after embedding), where dim 2 is the embedding space of size N
    - eps: epsilon value for gaussian kernel, a strictly positive Float64
    - m̂: estimate value of dimension x2
    - D: distance matrix, with distance info given by matrix N

    Keyword arguments
    =================
    - NN: nearest neighbors used
    - sym: boolean for optional operator symmetrization

"""
function sparseW_sepband(X::Matrix{Float64}, eps::Float64, m̂::Float64, 
     D::Matrix{Float64}, N::Matrix{Integer}; NN::Integer = 0, sym::Bool = true )
   # get distances
   # D, N = distNN(X, NN, usenorm = usenorm)
   
   nT = size(X, 1)
   if NN == 0
       NN = nT
   end

   point_density, _  = est_ind_bandwidth(D, NN, nT)
   m = m̂ / 2

   W = zeros(Float64, nT, nT)

   for i = 1:nT
       for j = 1:NN
           k = N[i,j]
           if i != k
                W[i, k] = sepbw_kernel(X[i,:], X[k,:], point_density[i], point_density[k], m, γ = eps)
           else
                W[i, k] = 1
           end
       end
   end

   if sym
       W = sym_M(W)
   end

   return W
end


"""
    normW(X::Matrix{Float64})

    normalizes sparse kernel matrix as described in https://doi.org/10.1038/s41467-021-26357-x 

    Arguments
    =================
    - X: square sparse kernel matrix

"""
function normW(X::Matrix{Float64})
    nX = size(X, 1)

    D = sum(X, dims = 2)
    D = D[:]
    Dinv = diagm(D.^(-1))
    
    S = zero(D)
    for i = 1:nX
        S[i] = sum(X[i,:] ./ D)
    end
    Sneghalf = diagm(S.^(-1/2))

    K̂ = Dinv * X * Sneghalf
    K̃ = K̂ * (K̂')

    return K̃
end

"""
    computeDiffusionEig(K::Matrix{Float64}, L::Integer = 0)

    computes L diffusion eigenfunctions from normalized kernel matrix K

    Arguments
    =================
    - K: normalized sparse kernel matrix
    - L: number of diffusion eigenfunctions to keep

"""
function computeDiffusionEig(K::Matrix{Float64}, L::Integer = 0)
    # TODO: test
    η, φ = eigen(K, sortby = x -> -real(x))
    if L > 0
        η = η[1:L]
        φ = φ[:,1:L]
    end
    w = φ[:,1].^2
    normφ = φ ./ φ[1,1]
    return η, normφ, w
end


"""
    diffSVD(φ::Matrix{Float64}, μ::Vector{Float64}, X::Matrix{Float64})

    UNTESTED 

    linear operator components and singular value decomposition
    as in algorithm that starts on line 15 of supplement of 10.1073/pnas.1118984109

    Arguments
    =================
    - φ: diffusion eigenfunctions of size s × L 
    - μ: vector of size s that are the wieghts from the normW function
    - X: original data (after embedding), where dim 2 is the embedding space of size N

"""
function diffSVD(φ::Matrix{Float64}, μ::Vector{Float64}, X::Matrix{Float64})

    s, L =  size(φ) # number of samples for which temporal modes are computed, number of Laplacian eigenfunctions
    N = size(X, 2) # embedding space dimension

    A = zeros(Float64, N, L)
    for j = 1:L
        for i = 1:N
            A[i,j] = sum(X[i, 2:(s + 1)] .* μ[1:s] .* φ[1:s, j])
        end
    end

    u, σ, v = svd(A)

    v2 = zeros(Float64, size(u))

    for k = 1:L
        for i = 1:s
            v2[i,k] = φ[i, 1:L] .* v[1:l, k]
        end
    end

    return u, σ, v2 
end


"""
    diffProjection(u::Matrix{Float64}, σ::Matrix{Float64}, v::Matrix{Float64};
    s::Integer, L::Integer, m::Integer, q::Integer)

    UNTESTED 

    projects diffusions eigenfunctions to physical space using linear op components from SVD
    (i.e. results of diffSVD), from appendix of 10.1073/pnas.1118984109

    Arguments
    =================
    - u, σ, v: from diffSVD(), i.e. [u, σ, v] = diffSVD()

    Keyword arguments
    =================
    - s: number of samples for which temporal modes are computed
    - L: number of computed diffusion eigenfunctions
    - m: physical space dimension
    - q: number of delays (no delays means q = 1)

"""
function diffProjection(u::Matrix{Float64}, σ::Matrix{Float64}, v::Matrix{Float64};
     s::Integer, L::Integer, m::Integer, q::Integer)

    X_proj = zeros(Float64, L, m, s)
    
    for k = 1:L
        for j = 1:s
            qprime = min(q, s - j + 1)

            for i = 1:qprime
                i1 = (i - 1) * m + 1
                i2 = i*m
                X_proj[k,1:m,j] = X_proj[k,1:m,j] + u[i1:i2, k] * σ[k] * v[j + i - 1, k]
            end

            X_proj[1:m, j] ./= qprime
        end
    end

    return X_proj
end


"""
    projectDiffEigs(φ::Matrix{Float64}, μ::Vector{Float64}, X::Matrix{Float64}, q::Integer = 1)

    UNTESTED

    computes projection of diffusion eigenvalues to physical space,
    from 10.1073/pnas.1118984109

    Arguments
    =================
    - φ: takes diffusion eigenfunctions of size s × L 
    - μ: vector of size s that are the wieghts from the normW function
    - X: original data (after embedding), where dim 2 is the embedding space of size N

    Keyword arguments
    =================
    - q: number of delays (no delays means q = 1)

"""
function projectDiffEig(φ::Matrix{Float64}, μ::Vector{Float64}, X::Matrix{Float64}; q::Integer = 1)
    s, L =  size(φ) # number of samples for which temporal modes are computed, number of Laplacian eigenfunctions
    N = size(X, 2)

    u, σ, v = diffSVD(φ, μ, X)
    X_proj = diffProjection(u, σ, v; s = s, L = L, m = N, q = q)

    return X_proj
end

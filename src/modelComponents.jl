export 
    tune_bandwidth,
    sparseW_mb,
    normW,
    computeDiffusionEig,
    normDiffEig,
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
    ris = sum(Dsq, dims = 2) ./ (NN - 1)

    nE = length(epsls)
    eps_sum = zeros(Float64, nE)
    
    Kel_sum = 0 
    for eee in 1:nE
        eps = epsls[eee]
        for m = 1:nT
            for n = 1:NN
                k = DN[m,n]
                Kel_sum += exp(-1*Dsq[m,n] / (eps * ris[m] * ris[k])^2)
                
            end
        end
        eps_sum[eee] = Kel_sum
    end

    eps_sum_l = log.(eps_sum)
    epsls_l = log.(epsls)

    eps_sum_prime = (eps_sum_l[2:end] - eps_sum_l[1:(end - 1)]) ./ (epsls_l[2:end] - epsls_l[1:(end - 1)])
    best_eps = epsls[argmax(eps_sum_prime)]

    return best_eps, eps_sum_prime
end

"""
    sparseW_mb(X::Matrix{Float64}, eps::Float64;
     NN::Integer = 0, usenorm::Function = norm, sym::Bool = true)

    computes sparse kernel matrix W with gaussian multiple bandwidth kernel from given ϵ

    Arguments
    =================
    - X: original data (after embedding), where dim 2 is the embedding space of size N
    - eps: epsilon value for gaussian kernel, a strictly positive Float64

    Keyword arguments
    =================
    - NN: nearest neighbors used
    - usenorm: function to compute distances with, defaults to l2
    - sym: boolean for optional operator symmetrization

"""
function sparseW_mb(X::Matrix{Float64}, eps::Float64;
    NN::Integer = 0, usenorm::Function = norm, sym::Bool = true )
   # get distances
   _, N = distNN(X[1:(end - 1),:], NN, usenorm = usenorm)
   
   nT = size(X, 1) - 1
   if NN == 0
       NN = nT
   end

   W = zeros(Float64, nT, nT)

   for i = 1:nT
       for j = 1:NN
           k = N[i,j]
           if i != k
                W[i, k] = varbandwidth_kernel(X[i + 1,:], X[k + 1,:], X[i,:], X[k,:], γ = eps)
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

function sparseW_mb_2(X::Matrix{Float64}, eps::Float64, dt::Float64;
    NN::Integer = 0, usenorm::Function = norm, sym::Bool = true )
   # get distances
   _, N = distNN(X[1:(end - 1),:], NN, usenorm = usenorm)
   
   nT = size(X, 1) - 1
   if NN == 0
       NN = nT
   end

   W = zeros(Float64, nT, nT)

   for i = 1:nT
       for j = 1:NN
           k = N[i,j]
           if i != k
                W[i, k] = varbandwidth_kernel2(X[i + 1,:], X[k + 1,:], X[i,:], X[k,:], dt, γ = eps)
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
    sparseW_mb2(X::Matrix{Float64}, eps::Float64;
     NN::Integer = 0, usenorm::Function = norm)

    computes sparse kernel matrix W with gaussian multiple bandwidth kernel from given ϵ

    Arguments
    =================
    - X: original data (after embedding), where dim 2 is the embedding space of size N
    - eps: epsilon value for gaussian kernel, a strictly positive Float64

    Keyword arguments
    =================
    - NN: nearest neighbors used
    - usenorm: function to compute distances with, defaults to l2
    - sym: boolean for optional operator symmetrization

"""
function sparseW_mb2(X::Matrix{Float64}, eps::Float64;
    NN::Integer = 0, usenorm::Function = norm, sym::Bool = true)
   # get distances
   _, N = distNN(X[1:(end - 1),:], NN, usenorm = usenorm)
   
   nT = size(X, 1) - 1
   if NN == 0
       NN = nT
   end

   W = zeros(Float64, nT, nT)

   for i = 1:nT
       for j = 1:NN
           k = N[i,j]
           if i != k
                W[i, k] = varbandwidth_kernel(X[i + 1,:], X[k + 1,:], X[i,:], X[k,:], γ = eps) / NN
           else
                W[i, k] = 1 / NN
           end
       end
   end

   return W
end


"""
    normW(X::Matrix{Float64})

    normalizes sparse kernel matrix as described in 10.1016/j.acha.2017.09.001

    Arguments
    =================
    - X: square sparse kernel matrix

"""
function normW(X::Matrix{Float64})
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

"""
    normW(X::Matrix{Float64})

    normalizes sparse kernel matrix as described in https://doi.org/10.1038/s41467-021-26357-x (instead?)

    Arguments
    =================
    - X: square sparse kernel matrix

"""
function normW2(X::Matrix{Float64})
    nX = size(X, 1)

    D = sum(X, dims = 2)
    D = D[:]
    Dinv = diagm(D.^(-1))
    
    S = zero(D)
    for i = 1:nX
        S[i] = sum(X[i,:] ./ D)
    end
    Sneghalf = diagm(S.^(-1/2))

    Khat = Dinv * X * Sneghalf
    Ktilde = Khat * (Khat')

    return Ktilde
end

"""
    normW(X::Matrix{Float64})

    normalizes sparse kernel matrix as described in https://doi.org/10.1038/s41467-021-26357-x (instead?)

    Arguments
    =================
    - X: square sparse kernel matrix

"""
function normW3(X::Matrix{Float64})
    nX = size(X, 1)

    D = sum(X, dims = 2)
    D = D[:]
    Dinv = diagm(D.^(-1))
    
    S = zero(D)
    S = sum(X * Dinv, dims = 2)[:]
    print(size(S))
    # for i = 1:nX
    #     S[i] = sum(X[i,:] ./ D)
    # end
    # Sneghalf = diagm(S.^(-1/2))


    Ktilde = Dinv * X * diagm(S.^(-1/2))

    return Ktilde
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
    normDiffEig(φ::Matrix{Float64}, L::Integer)

    normalize diffusion eigenfunctions as in 10.1016/j.acha.2017.09.001
    
    Arguments
    =================
    - φ: diffusion eigenfunctions of size s × L
    - κ: diffusion eigenvalues as computed in computeDiffusionEig()

"""
function normDiffEig(φ::Matrix{Float64}, κ::Vector{Float64})
    s, L = size(φ)
    normφ = zeros(Float64, s, L)

    for k = 1:L
        normφ[:,k] = φ[:,k] ./ norm(φ[:,k])
    end
    normφ = normφ ./ mean(φ[:,1]) # normalized eigenfunctions
    η = log.(κ) ./ log.(κ[1]) # normalized eigenvalues
    w = φ[1,:].^2 # inner product weights

    return normφ, η, w
end


"""
    diffSVD(φ::Matrix{Float64}, μ::Vector{Float64}, X::Matrix{Float64})

    linear operator components and singular value decomposition
    as in algorithm that starts on line 15 of supplement of 10.1073/pnas.1118984109
    TODO: test


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

    projects diffusions eigenfunctions to physical space using linear op components from SVD
    (i.e. results of diffSVD), from appendix of 10.1073/pnas.1118984109
    TODO: test

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

    computes projection of diffusion eigenvalues to physical space,
    from 10.1073/pnas.1118984109
    TODO: test

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
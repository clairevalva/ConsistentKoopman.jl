using LinearAlgebra

function rbf_kernel(x::Vector{Float64},y::Vector{Float64}, σ = 1)
    return exp.(-norm(x .- y)^2 / σ)
end

function simplevel_kernel(x::Vector{Float64}, y::Vector{Float64},
    xpre::Vector{Float64}, ypre::Vector{Float64}, eps; usenorm::Function = norm)
    # from 10.1073/pnas.1118984109

    vx = usenorm(x .- xpre)
    vy = usenorm(y .- ypre)
    distxy = usenorm(x .- y)

    divC = vx*vy*eps
    return exp(-1*distxy / divC)
end

function sepBandwidth(x::Vector{Float64}, y::Vector{Float64},
     xpre::Vector{Float64}, ypre::Vector{Float64}; ζ = 0.995)
    # from appendix in Froyland paper
    # TO DO: TEST 
    vx = x .- xpre
    vy = y .- ypre

    vx = vx / norm(vx)
    vy = vy / norm(vy)

    diffxy = x .- y
    diffxy = diffxy / norm(diffxy)
    
    cθ1 = -vx' * diffxy
    cθ2 = vy' * diffxy

    return sqrt((1 - ζ*cθ1) * (1 - ζ*cθ2))
end

function varbandwidth_kernel(x::Vector{Float64}, y::Vector{Float64},
    xpre::Vector{Float64}, ypre::Vector{Float64}; γ = 33, ζ = 0.995)
    # TODO: test
    bandf = sepBandwidth(x,y, xpre, ypre, ζ = ζ)
    diffxy = norm(x .- y)
    return exp.((diffxy /  (bandf * γ))^2)
end

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

# if I want to tune bandwidth
function tune_bandwidth(D::Matrix{Float64}, DN::Matrix{Integer}, 
    NN::Integer, nT::Integer, epsls::Vector{Float64})
    # from 10.1016/j.acha.2017.09.001
    # probably cheating a bit by only taking the sum of the NN nearest neighbors
    # hoping it doesn't matter too much

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
                Kel_sum += exp(-1*Dsq[m,n] / (eps * ris[m] * ris[k]))
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

function sparseW_bs(X::Matrix{Float64}, epsls::Vector{Float64};
     NN::Integer = 0, usenorm::Function = norm, sym::Bool = true )
    # get distances
    D, N = distNN(X[1:(end - 1),:], NN, usenorm = usenorm)
    
    nT = size(X, 1) - 1
    if NN == 0
        NN = nT
    end

    # tune eps
    eps = tune_bandwidth(D, N, NN, nT, epsls)
    W = zeros(Float64, nT, nT)

    for i = 1:nT
        for j = 1:NN
            k = N[i,j]
            W[i, k] = varbandwidth_kernel(X[i + 1,:], X[k + 1,:], X[i,:], X[k,:], γ = eps)
        end
    end

    if sym
        W = sym_M(W)
    end

    return W
end

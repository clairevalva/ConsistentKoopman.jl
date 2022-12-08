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
    return exp.(-1*(diffxy /  (bandf * γ))^2)
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

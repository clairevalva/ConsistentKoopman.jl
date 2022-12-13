export 
    rbf_kernel,
    simplevel_kernel,
    sepBandwidth

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
    # from appendix in Froyland paper: 10.1038/s41467-021-26357-x
    
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
    bandf = sepBandwidth(x,y, xpre, ypre, ζ = ζ)
    diffxy = norm(x .- y)
    return exp.(-1*(diffxy /  (bandf * γ))^2)
end
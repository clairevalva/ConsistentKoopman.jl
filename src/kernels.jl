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

    # return exp.(-1*(diffxy * bandf / γ)^2)
    return exp.(-1*(diffxy * bandf / γ)^2)
end

function varbandwidth_kernel_cone(x::Vector{Float64}, y::Vector{Float64},
    xpre::Vector{Float64}, ypre::Vector{Float64}, dt::Float64; γ = 33, ζ = 0.995)
    bandf = sepBandwidth(x,y, xpre, ypre, ζ = ζ)
    diffxy = norm(x .- y)
    vel1 = norm((x - xpre) ./ dt)
    vel2 = norm((y - ypre) ./ dt)
    
    # return exp.(-1*(diffxy * bandf / γ)^2)
    return exp.(-1*(diffxy^2 * bandf / (γ * vel1 * vel2)))
end

function varbandwidth_kernel_sep(x::Vector{Float64}, y::Vector{Float64}, px::Float64, py::Float64, m::Float64; γ = 33)
    diffxy = norm(x .- y)
    σ_sq = (px * py)^(-1/m)
    
    return exp.(-1*(diffxy * σ_sq / γ)^2)
end

"""
est_ind_bandwidth(D::Matrix{Float64}, NN::Integer, nT::Integer)

    https://doi.org/10.1016/j.acha.2015.01.001

    Arguments
    =================
    - D: precomputed distances for nearest neighbors NN (from distNN)
    - NN: number of nearest neighbors
    - nT: total time of processed data

"""
function est_ind_bandwidth(D::Matrix{Float64}, NN::Integer, nT::Integer)
    
    Dsq = D.^2
    point_density = zeros(nT)

    for m = 1:nT
        point_density[m] = (sum(Dsq[m, :]) / (NN - 1))^0.5
    end

    ϵ_0_half = sum(point_density) / nT 

    return point_density, ϵ_0_half
end


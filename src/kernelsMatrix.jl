export 
    makeSϵ,
    tuneBandwidth,
    coneBandwidths,
    sepBandwidths,
    makeW


"""
    makeSϵ(D::AbstractMatrix, σ)

    makes function S from distances D and bandwidths σ for parameter tuning (can pass σ = 1 if needed )
    
        The following assume all kernels are of the form:
            kᵧ(x, y) = exp(- ‖ x - y ‖^2 / (γ^2 σ^2(x, y)))
            and that we have already computed the distances D, i.e. D[xi, xj] = ‖ xi - xj ‖

    Arguments
    =================
    - D: distances matrix of size nT × nT, currently assumes no nearest neighbors
    - σ: either matrix of size nT × nT or integer bandwidth

"""
function makeSϵ(D::AbstractMatrix, σ)
    function estSϵ(γ)
        γ = exp(γ)
        kentries = exp.(-1*D.^2 ./ (γ^2 * σ.^2))
        S = sum(kentries)
        return log(S)
    end
end

"""
    tuneBandwidth(D::AbstractMatrix, σ, testγ::AbstractVector)

    tunes bandwith from distances D, bandwidths σ to tuning parameter

        The following assume all kernels are of the form:
            kᵧ(x, y) = exp(- ‖ x - y ‖^2 / (γ^2 σ^2(x, y)))
            and that we have already computed the distances D, i.e. D[xi, xj] = ‖ xi - xj ‖        

    Arguments
    =================
    - D: distances matrix of size nT × nT, currently assumes no nearest neighbors
    - σ: either matrix of size nT × nT or integer bandwidth
    - testγ: vector of parameters to choose between for tuning

"""
function tuneBandwidth(D::AbstractMatrix, σ, testγ::AbstractVector)
    # make sure testγ is log
    S = makeSϵ(D, σ)
    S_grad(γ) = ForwardDiff.derivative(S, γ)
    Sgs = S_grad.(testγ)

    if !isfinite(sum(Sgs))
        println("seems like an overflow error... σ too big?")
        N = findfirst(x -> isnan(x), Sgs)
    
        Sgs = Sgs[1:N-1]
    end

    bestj = argmax(Sgs)
    bestϵ = exp.(testγ[bestj])
    m̂ = Sgs[bestj]

    return bestϵ, m̂
end

"""
    coneBandwidths(X::AbstractMatrix, D::AbstractMatrix; ζ = 0.995, returnall = false)

    from appendix in Froyland paper: 10.1038/s41467-021-26357-x, this is \frac{1}{σ{x_i, x_j}}
    conebw from other file is still an ok alternative, tested to be the same

    makes bandwidths matrix for final size (nT - 1) × (nT - 1)
    
    Arguments
    =================
    - X: original data matrix of size nT × nT, 
    - D: distances matrix of size nT × nT, currently assumes no nearest neighbors
    - ζ = 0.995 tuned parameter
    - returnall: keep as false unless for debugging, but currently matches outputs so unnecessary
"""
function coneBandwidths(X::AbstractMatrix, D::AbstractMatrix; ζ = 0.995, returnall = false)
    _, nT = size(X)
    cθ1 = zeros(nT - 1, nT - 1)
    cθ2 = zeros(nT - 1, nT - 1)

    vx = X[:, 2:nT] .- X[:, 1:(nT - 1)]
    vx = vx ./ norm.(eachcol(vx), 2)'
    

    for i = 1:(nT - 1)
        for j = 1:(i-1)
            diffxy = (X[:, i + 1] - X[:, j + 1]) / D[i + 1, j + 1]
            cθ1[i, j] = -1*vx[:,i]' * diffxy
            cθ2[i, j] = vx[:,j]' * diffxy
        end
    end

    cθ1 .+= cθ1'
    cθ2 .+= cθ2'

    σinv = sqrt.((1 .- ζ*cθ1) .* (1 .- ζ*cθ2)).^-1
    if returnall
        return vx, cθ1, cθ2, diffs, σinv
    else
        return σinv
    end
end

"""
    sepBandwidths(D::AbstractMatrix, m::Real)

    makes seperable bandwidths S from distances D and bandwidths σ for parameter tuning (can pass σ = 1 if needed )

    Arguments
    =================
    - D: distances matrix of size nT × nT, currently assumes no nearest neighbors
    - m: space dimension parameter, often estimated from the derivative of the tuned bandwidth

"""
function sepBandwidths(D::AbstractMatrix, m::Real)
    nT = size(D, 1)
    point_density = (sum(D.^2, dims = 2 ) / (nT - 1)) .^ 0.5

    σ = sqrt.((point_density * point_density') .^ (1 / m))
    return σ 
end

"""
    sepBandwidths(D::AbstractMatrix, NN::Integer, m::Real)

    makes seperable bandwidths S from distances D and bandwidths σ for parameter tuning (can pass σ = 1 if needed )
    with integer parameter

    Arguments
    =================
    - D: distances matrix of size nT × nT, currently assumes no nearest neighbors
    - NN: nearest neighbors for better pt density
    - m: space dimension parameter, often estimated from the derivative of the tuned bandwidth

"""
function sepBandwidths(D::AbstractMatrix, NN::Integer, m::Real)
    nT = size(D, 1)
    point_density = (sum(D.^2, dims = 2 ) / (NN - 1)) .^ 0.5

    σ = sqrt.((point_density * point_density') .^ (1 / m))
    return σ 
end


"""
    makeW(D::AbstractMatrix, γ::Real, σ)

    makes seperable bandwidths S from distances D and bandwidths σ for parameter tuning (can pass σ = 1 if needed )

    Arguments
    =================
    - D: distances matrix of size nT × nT, currently assumes no nearest neighbors
    - γ: tuned bandwidth
    - σ: individual bandwidth, matrix the same size as D, pass σ = 1 for gaussian rbf

"""
function makeW(D::AbstractMatrix, γ::Real, σ)
    K = exp.(-1*D.^2 ./ (γ^2 * σ.^2))
    return K
end
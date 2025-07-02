# for my ease, these have replicate function definitions, do not incorporate as is 

function conebw(x::Vector{Float64}, y::Vector{Float64},
     xpre::Vector{Float64}, ypre::Vector{Float64}; ζ = 0.995)
    # from appendix in Froyland paper: 10.1038/s41467-021-26357-x, this is \frac{1}{σ{x_i, x_j}}
    
    vx = x .- xpre
    vy = y .- ypre

    vx = vx / norm(vx, 2)
    vy = vy / norm(vy, 2)

    diffxy = x .- y
    diffxy = diffxy / norm(diffxy, 2)
    
    cθ1 = -vx' * diffxy
    cθ2 = vy' * diffxy

    return sqrt((1 - ζ*cθ1) * (1 - ζ*cθ2))
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


function makeNLSAkernel(params::paramsNLSA, kernel_choice; m::Real = 1)
    # m is for seperable bandwidths
    X = params.srcdata
    NN = params.NN
    candidate_ϵs = params.candidate_ϵs
    nDiff = params.nDiff

    nT = size(X, 2)

    println("computing distances")
    D, DN = distNN(X, NN)

    if iszero(NN)
        NN_bw = nT
    else
        NN_bw = NN
        
        println("symmetric NNs")
        DN = .!iszero.(DN + DN')
        D[.!DN] .= 0
    end


    println("computing ind bandwidth (σ)")
    if kernel_choice == "gaussian"
        σ = 1
    elseif kernel_choice == "cone"
        σ = coneBandwidths(X, D)
        X = X[:, 2:end]
        D = D[2:end, 2:end]
        nT -= 1
    else
        error("kernel not implemented")
    end

    println("computing bandwidth (ϵ)")
    bw, m = tuneBandwidth(D, σ, candidate_ϵs)

    println("make kernel matrix")
    W = makeW(D, bw, σ)
    
    println("normW")
    P = normW(W)

    if kernel_choice == "cone"
        bw_func = conebw #conebw(x::Vector{Float64}, y::Vector{Float64},xpre::Vector{Float64}, ypre::Vector{Float64}; ζ = 0.995)
        
        


        
    else
        error("kernel not implemented")
    end

    println("norm kernel (slow)")
    



    println("NLSA eigendecomposition")
    κ, φ, w = computeDiffusionEig(P, nDiff)

    if kernel_choice == "seperable"
        return eigsNLSA(params, κ, φ, w), m̂, bw
    else 
        return eigsNLSA(params, κ, φ, w)
    end
end
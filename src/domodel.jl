export 
    paramsNLSA,
    NLSAeigs,
    paramsKoop,
    makeParamsKoop,
    eigsKoop,
    doKoopman,
    doNLSA,
    doNLSAMatrix

struct paramsNLSA
    srcdata::Matrix{Float64}
    srcdt::Float64
    NN::Int64
    candidate_ϵs::Vector{Float64}
    nDiff::Int64
end

struct eigsNLSA
    params::paramsNLSA
    κ::Vector{Float64}
    φ::Matrix{Float64}
    w::Vector{Float64}
end

function doNLSA(params::paramsNLSA, kernel_choice)
    X = params.srcdata
    NN = params.NN
    candidate_ϵs = params.candidate_ϵs
    nDiff = params.nDiff

    nT = size(X, 2)

    if NN == 0
        NN_bw = nT
    else
        NN_bw = NN
    end

    print("computing distances")
    D, DN = distNN(X, NN)
    print("computing bandwidth")
    useϵ, m̂ = tune_bandwidth(D, DN, NN_bw, nT, candidate_ϵs)
    print("sparseW")
    if kernel_choice == "cone"
        W = sparseW_cone(X, useϵ, m̂, D, DN, NN = NN, sym = true)
    else 
        W = sparseW_sepband(X, useϵ, m̂, D, DN, NN = NN, sym = true)
    end
    W = sparse(W)
    print("normW")
    P = normW(W)
    κ, φ, w = computeDiffusionEig(P, nDiff)

    return eigsNLSA(params, κ, φ, w)
end

function doNLSAMatrix(params::paramsNLSA, kernel_choice; m::Real = 1)
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
    elseif kernel_choice == "seperable"
        if isnan(m) & iszero(NN)
            bw, m̂ = tuneBandwidth(D, 1, candidate_ϵs)
            σ = sepBandwidths(D, m̂ * 2) 
        elseif isnan(m)
                bw, m̂ = tuneBandwidth(D, 1, candidate_ϵs)
                σ = sepBandwidths(D, NN, m̂ * 2) 
        elseif iszero(NN)
            σ = sepBandwidths(D, m)
        else
            σ = sepBandwidths(D, NN, m)
        end
    else
        error("kernel not implemented")
    end

    println("computing bandwidth (ϵ)")
    bw, m = tuneBandwidth(D, σ, candidate_ϵs)

    println("make kernel matrix")
    W = makeW(D, bw, σ)
    
    println("normW")
    P = normW(W)

    println("NLSA eigendecomposition")
    κ, φ, w = computeDiffusionEig(P, nDiff)

    if kernel_choice == "seperable"
        return eigsNLSA(params, κ, φ, w), m̂, bw
    else 
        return eigsNLSA(params, κ, φ, w)
    end
end


struct paramsKoop
    srcdata::Matrix{Float64}
    srcdt::Float64
    NN::Int64
    candidate_ϵs::Vector{Float64}
    nDiff::Int64

    nKoop::Int64
    mKoop::Int64
    z::Float64
    τ::Float64

    κ::Vector{Float64}
    φ::Matrix{Float64}
    w::Vector{Float64}
end

struct eigsKoop
    params::paramsKoop
    ω::Vector{Float64}
    ζ::Matrix{ComplexF64}
    c::Matrix{ComplexF64}
end

function makeParamsKoop(diffresults::eigsNLSA, nKoop::Int64, mKoop::Int64, z::Float64, τ::Float64)
    diffparams = diffresults.params
    srcdata = diffparams.srcdata
    srcdt = diffparams.srcdt
    NN = diffparams.NN
    candidate_ϵs = diffparams.candidate_ϵs
    nDiff = diffparams.nDiff

    newparams = paramsKoop(srcdata, srcdt, NN, candidate_ϵs, nDiff, nKoop, mKoop, z, τ, diffresults.κ, diffresults.φ, diffresults.w)
    return newparams
end

function doKoopman(params::paramsKoop)
    φ = params.φ
    κ = params.κ
    w = params.w
    
    dt = params.srcdt
    τ = params.τ
    z = params.z
    nKoop = params.nKoop
    mKoop = params.mKoop

    φ_plus = posfilter(φ)
    G = Gtau(κ, τ)[1:nKoop, 1:nKoop]
    Rz = resolventop_power(φ_plus, w, 50, dt, z)[1:nKoop, 1:nKoop]
    ω, ζ, c = computeSeigs(Rz, G, z, nKoop, mKoop, φ)

    return eigsKoop(params, ω, ζ, c), Rz
end


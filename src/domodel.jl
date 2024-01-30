export 
    paramsNLSA,
    NLSAeigs,
    paramsKoop,
    makeParamsKoop,
    eigsKoop,
    doKoopman,
    doNLSA

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
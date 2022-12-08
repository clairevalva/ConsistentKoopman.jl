mutable struct NLSAmodel
    srcData::Array{Float64,2} # data should be size nT × nD 
    dt::Float64 # time resolution of data
    nDelays::Int # number of delays for embedding

    kernel::Function # kernel function of use
    kernelvel::Bool # does the kernel need more info? (i.e. is it computing velocities?)
    kernelargs::Array # extra arguments to the kernel
    nφ::Int
end

function doNLSA(model::NLSAmodel)
    # assumes that the kernel is symmetric here
    X = model.srcData
    Xemb = delayembed(X', model.nDelays)' # this takes array in the wrong shape
    
    if model.kernelvel
        ker(x,y, xmin, ymin) = model.kernel(x, y, xmin, ymin, model.kernelargs...)
        Kdists = computeKdists_vel(ker, Xemb)
    else
        ker(x,y) = model.kernel(x, y, model.kernelargs...)
        Kdists = computeKdists(ker, Xemb)
    end

    K = constructK(Kdists)
    η, φ = computeDiffusionEig(K, model.nφ)  

    return Xemb, η, φ
end



function computeDiffusionEig(K::Matrix{Float64}, L::Integer = 0)
    # TODO: test
    η, φ = eigen(K, sortby = x -> -real(x))
    if L > 0
        η = η[1:L]
        φ = φ[:,1:L]
    end

    return η, φ
end
export 
Gtau
posfilter
koopmanop
polarfz
makeζ

using FFTW
# TO DO: test everything!
"""
    Gtau(η::Vector{Float64}, τ::Float64)

    construction of Gtau matrix where τ ↦ τ/2

    Arguments
    =================
    - η: Diffusion eigenvalues (NLSA eigenvalues), in usual descending order
    - τ: value of τ to use, must be greater than zero

"""
function Gtau(η::Vector{Float64}, τ::Float64)
    
    if τ ≤ 0
        error("τ must be greater than 0")
    else
        λ = η.^(-1) .- 1
        λ = λ ./ λ[2]
        λ_τ = exp.(-1*λ*τ)
        G = diagm(λ_τ.^(1/2))
    end

    return G
end

"""
    posfilter(φ::Matrix{Float64})

    filters values of fft corresponding to negative frequencies to zero

    Arguments
    =================
    - φ: diffusion eigenfunctions of size s × L 
"""
function posfilter(φ::Matrix{Float64})
    s = size(φ, 1)
    L = size(φ, 2)
    f = fftfreq(s)
    fbool = (f < 0)

    φplus = 1j .* zero(φ)

    for r = 1:L
        ffted = fft(φ[:,r])
        ffted[fbool] .= 0
        φplus[:,r] = ifft(ffted)
    end

    return φplus
end

"""
    koopmanop(shifts::Vector{Int64}, φ::Matrix{Float64}, w::Vector{Float64})

    compute approximation of Koopman operator using a shift matrix on a basis of observables (φ)

    Arguments
    =================
    - shifts: a vector of shifts to apply, if only one give as a vector, i.e. [3] because I am very lazy 
    - φ: diffusion eigenfunctions of size s × L 
    - w: inner product weight array of size s × 1
"""
function koopmanop(shifts::Vector{Int64}, φ::Matrix{Float64}, w::Vector{Float64})
    nQ = length(shifts)
    _ , L = size(φ)
    U = zeros(nQ, L, L)

    for j = 1:nQ
        q = shifts[j]

        if q < 0
            b = abs(q)
            U[j,:,:] = φ[b + 1:end, :]' * (φ[1:end - b, :] .* w[b + 1 : end])
        else
            U[j,:,:] = φ[1:end - q, :]' * (φ[q + 1:end, :] .* w[1 : end - q])
        end
    end

    return U
end


"""
resolventop_power(φ::Matrix{Float64}, w::Vector{Float64}, Tf::Int64, dt::Float64, z::Float64; batchN = 1)

    compute approximation of Koopman operator using a shift matrix on a basis of observables (φ)

    Arguments
    =================
    - φ: diffusion eigenfunctions of size s × L 
    - w: inner product weight array of size s × 1
    - Tf: final time to integrate to
    - dt: data spacing
    - z: real number to evaluate resolvent at
    
    Keyword arguments
    =================
    - batchN: number of batches in which to run quadrature
"""
function resolventop_power(φ::Matrix{Float64}, w::Vector{Float64}, Tf::Int64, dt::Float64, z::Float64; batchN = 1)

    U = koopmanop([1], φ, w)[1,:,:]

    _ , L = size(φ)
    ts = 0:dt:Tf
    nT = length(ts)
    qs = 0:(nT - 1)
    lenBatch = ceil(Int64, nT / batchN)
    

    I = zero(U)
    for j = 1:batchN

        # get index for this batch
        if j == batchN
            a_st = (j - 1)*lenBatch + 1
            a_end = nT
        else
            a_st = (j - 1)*lenBatch + 1
            a_end = j*lenBatch
        end

        ttt = ts[a_st:a_end]
        qqq = qs[a_st:a_end]
        eee = exp.(-1*z*ttt)
        lb = length(qqq)
        
        Ut = zeros(lb, L, L)
        for k = 1:lb
            Ut[k,:,:] = U^qqq[k] .* eee[k]
        end
        
        # simps rule code from from here: https://discourse.julialang.org/t/simpsons-rule/84114/6
        I += dt/3 .* (Ut[1,:,:] + 2*sum(Ut[3:2:end-2, :, :], dims = 1)[1,:,:] + 4*sum(Ut[2:2:end, :, :], dims = 1)[1,:,:] + Ut[end, :, :])
    end

    return I
end


function polarfz(evals::Vector{Float64}, z::Float64)
    rrr = 1 / (2 * z)
    xdiv = abs.(evals/2)
    argu = xdiv ./ rrr
    theta = asin.(argu)
    alpha = (pi .- 2*theta) ./ 2
    lambda = evals .* exp.(1im * alpha)

    return lambda
end

function makeζ(keigvec::Matrix{Float64}, diffeigvec::Matrix{Float64})
    ζ = diffeigvec * keigvec;
    return ζ
end
# function eigorder(zeta::Matrix{Float64}, ωs::Vector{Float64}, dt::Float64, T::Int64)
#     L = length(ωs)

#     actACs = zeros(L, T + 1)
#     goalACs = zero(actACs)
#     lags = 0:T
#     ts = dt*lags

#     for j = 1:L
#         cf = autocor(zeta, lags)
#     end

# end
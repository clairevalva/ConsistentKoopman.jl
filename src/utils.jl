export
    delayembed,
    distNN,
    sym_M,
    sortautocorr


"""
    delayembed(X, numdelays::Int)

    delay embedding for data, should return
    Xemb: ((numdelays)⋅n) × (m - numdelays - 1) data array

    # '' appears to work for the following:
    # testMat = [1 2 3 4 5 6 ; 7 8 9 10 11 12 ; 13 14 15 16 17 18];
    # k = 0, 1, 2, 3
    # delayembed(testMat,k)
    # delayembed(testMat,0) == testMat '' 
    
    Arguments
    =================
    - X: n × m data array with n spatial points and m timepoints
    - numdelays: integer, number of delays (should be that q = 1 is no delays for consistency)

"""
function delayembed(X, numdelays::Int)
        numdelays -= 1
        if numdelays == 0
            return X
        end

        n = size(X,1);
        m = size(X,2);
        Xemb = zeros(typeof(X[1]), n*(1 + numdelays), m - numdelays);
        for del in 0:numdelays
            Xemb[del*n .+ (1:n), :] = X[:, (1 + del):(m - numdelays + del)];
        end
        
        return Xemb
end


"""
    distNN(X::Matrix{Float64}, NN::Integer = 0; usenorm::Function = norm)

    computes distances from matrix M, keeps distance (D) and indexing info (N) for NN nearest neighbors

    Arguments
    =================
    - X: data matrix of size nT × nD (time by spatial dim)
    - NN: positive nearest neighbors parameter, if 0 defaults to keeping all

    Keyword arguments
    =================
    - usenorm: specify the distance norm to use, defaults to l2 

"""
function distNN(X::Matrix{Float64}, NN::Integer = 0; usenorm::Function = norm)
    nT, _ = size(X)
    D = zeros(Float64, nT, nT)
    N = zeros(Integer, nT, nT)

    if NN == 0
        # if no nearest neighbors specified, keep all of them
        NN = nT
    end

    for i = 1:nT
        d = zeros(Float64, nT)
        for j = 1:nT
            d[j] = usenorm(X[i,:] .- X[j,:])
        end
        inds = sortperm(d)[1:NN]
        D[i,1:NN], N[i,1:NN] = d[inds], inds
    end
    # then have that D(i,j), d(N(j))
    return D, N
end

"""
    sym_M(M::Matrix{Float64})

    function to symmetrize sparse kernel matrix as computed in sparseW()

        on June 14 I reversed the equality, to chekc later
    Arguments
    =================
    - M: square sparse kernel matrix

"""
function sym_M(M::Matrix{Float64})
    N = size(M,1)
    # TO DO
    for i = 1:N
        for j = 2:i
            if (M[i,j] == 0) & (M[j,i] != 0)
                M[j,i] = M[i,j]
            elseif (M[i,j] != 0) & (M[j,i] == 0)
                M[i,j] = M[j,i]
            end
        end
    end

    return M
end

"""
    polar(A::Matrix{Float64})

    computes polar decomposition of a square matrix using svd
    Arguments
    =================
    - A: square matrix

"""
function polar(A::Matrix)
    U, S, V = svd(A)
    B = U * V'
    C = V * diagm(S) * V'

    return B, C # B unitary, C positive semi-def hermitian
end


"""
    crosscorrcomplex(x::Vector{ComplexF64}, y::Vector{ComplexF64}, m::Int64; normed = false)

    computes the cross correlation between *complex* vectors x and y (because the stats base function does NOT). For auto correlation, use x = y, and for normalization in that case set normed = true

    Arguments
    =================
    - x: complex vector of length L
    - y: complex vector of length L
    - m: integer strictly less than l, number of nLags
    - normed: whether or not to norm by the value when m = 0

"""
function crosscorrcomplex(x::Vector{ComplexF64}, y::Vector{ComplexF64}, m::Int64; normed = false)
    ac = sum(x[m + 1:end] .* conj.(y[1:(end - m)]))

    if normed
        ac = ac / sum(x .* conj.(y))
    end
    return ac
end

"""
    sortautocorr(ζ::Matrix{Float64})

    CHECKED THAT THIS WORKED ON SEPT 14, 2023

    computes polar decomposition of a square matrix using svd
    Arguments
    =================
    - A: square matrix

"""

function sortautocorr(ζ::Matrix{ComplexF64}, frequencies::Vector{Float64}, nLags::Int64, dt::Float64; Flim::Float64 = 30., returnall = false)
    L = size(ζ, 2)
    acs = 1im*zeros(nLags + 1, L)
    goalacs = 1im*zeros(nLags + 1, L)
    
    for j in 1:L
        goalacs[:, j] = exp.(-1im*frequencies[j]*(0:nLags)*dt)
        for lag in 0:nLags
            acs[lag + 1, j] = crosscorrcomplex(ζ[:,j], ζ[:,j], lag, normed = true)
        end
    end

    eps_t = real(1 .- abs.(goalacs .* acs))
    max_eps = maximum(eps_t, dims = 1)[1,:] + (abs.(frequencies) .> Flim)
    sortinds = sortperm(vec(max_eps))
    if returnall
        return max_eps, sortinds, acs, goalacs
    else
        return max_eps, sortinds
    end

end
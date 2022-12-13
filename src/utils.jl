export
    delayembed,
    distNN,
    sym_M


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
            elseif (M[i,j] != 0) & (M[j,i] == 0)
                M[j,i] = M[i,j]
            end
        end
    end

    return M
end
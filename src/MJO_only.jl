# do NLSA part only for MJO for compatibility with Nystrom extension

# X is any data
# NN = 0, for now I gueess 


function doNLSA(params::paramsNLSA)
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

    σ = coneBandwidths(X, D)
    X = X[:, 2:end] # X[:, 2:end]
    D = D[2:end, 2:end]
    nT -= 1

    println("computing bandwidth (ϵ)")
    bw, _ = tuneBandwidth(D, σ, candidate_ϵs)

    println("make kernel matrix")
    W = makeW(D, bw, σ)
    
    println("normW")
    P = normW(W)

    println("NLSA eigendecomposition")
    κ, φ, w = computeDiffusionEig(P, nDiff)

    println("normalized kernel func")

    k_evals, k_norm = makeNormKernel(X, nT, bw)

    return κ, φ, w, k_evals, k_norm
end

function normWPieces(X::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}})
    # AHHH
    D = sum(X, dims = 2)[:]
    Dinv = Diagonal(D.^(-1))
    Q = sum(X ./ (D'), dims = 2)[:]
    Qneghalf = Diagonal(Q.^(-1/2))

    K̂ = Dinv * X * Qneghalf
    

    return diag(Qneghalf), K̂
end


function makeNormKernel(W, X, nT, bw)
    Qneghalf, K̂ = normWPieces(W)
    
    # assemble initial kernel
    function k(x, y, xpre, ypre)
        D = euclidean(x, y)
        σ = ConsistentKoopman.conebw(x, y, xpre, ypre).^-1
        return exp.(-1 * D ./ (σ .^ 2 * bw .^ 2))
    end

    function d(y, ypre)
        d_sum = 0
        for k = 1:nT
            d_sum += k(y, X[:, k + 1], ypre, X[:, k])
        end

        return d_sum / nT
    end

    function khat(y, ypre, l)
        kval =  k(y, X[:, l + 1], ypre, X[:, l])
        dval = d(y, ypre)
        qval = Qneghalf[l]

        return kval / dval * qval
    end

    function k_norm(y, ypre, l)
        k_sum = 0
        for j = 1:nT
            khat_val = khat(y, ypre, j)
            k_sum += khat_val * K̂[j, l]
        end

        return k_sum / nT
    end

    function k_faster(y, ypre)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] += k(y, X[:, j + 1], ypre, X[:, j])
        end

        dval = sum(k_evals) / nT
        println(dval)
        khat_evals = (k_evals ./ dval) .* Qneghalf

        k_sum = zeros(nT)
        for l = 1:nT
            for j = 1:nT
                k_sum[l] += khat_evals[j] *  K̂[j, l]
            end
        end

        k_sum = k_sum / nT

        return k_sum
    end
    return k_faster, k_norm # TO DO: check that these coincide appropriately, but I think they should
end
function symDist(M)
    testequal = (M .== M')
    foundinds = findall(iszero, testequal)
    
    for entry in foundinds
        if entry[1] > entry[2]
            if M[entry] > 0.5
               M[entry[2], entry[1]] = true
            else
                M[entry] = true
            end

        end
    end

    return M
end

function normWPieces(X::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}})
    D = sum(X, dims = 2)[:]
    Dinv = Diagonal(D.^(-1))
    S = sum(X ./ (D'), dims = 2)[:]
    Sneghalf = Diagonal(S.^(-1/2))

    K̂ = Dinv * X * Sneghalf
    

    return diag(Sneghalf), K̂
end

function makeRbfKer(bw)
    # assemble initial kernel
    function k(x, y)
        D = euclidean(x, y)
        return exp.(-1 * D.^2 ./ (bw .^ 2))
    end
    return k
end

function makeNormKernel_rbf(W, X, nT, bw)
    Qneghalf, K̂ = normWPieces(W)
    
    k = makeRbfKer(bw)

    println(nT)
    function k_faster(y; verbose = false)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] = k(y, X[:, j])
        end
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) .* Qneghalf
        
        k_sum = zeros(nT)
        for l = 1:nT
            for j = 1:nT
                k_sum[l] += khat_evals[j] *  K̂[l, j] 
            end
        end

        k_sum = Float64.(k_sum) #/ nT
        if verbose
            return k_sum, k_evals, khat_evals, dval, Qneghalf
        else
            return k_sum
        end
    end

    return k_faster
end

function makeNormKernel_rbf(W, X, nT, bw, NN)
    if NN == 0
        return makeNormKernel_rbf(W, X, nT, bw)
    end
    Qneghalf, K̂ = normWPieces(W)
    
    k = makeRbfKer(bw)

    function k_faster(y; verbose = false)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] = k(y, X[:, j])
        end

        k_sort = sortperm(-1*k_evals)

        if verbose
            println("length eliminated: ", length(k_sort[NN + 1:end]))
            println("max elim: ", maximum(k_evals[k_sort[NN + 1:end]]))
            println("min elim: ", minimum(k_evals[k_sort[NN + 1:end]]))
        end

        k_evals[k_sort[NN + 1:end]] .= 0
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) .* Qneghalf
        
        k_sum = zeros(nT)
        for l = 1:nT
            for j = 1:nT
                k_sum[l] += khat_evals[j] *  K̂[l, j] 
            end
        end

        k_sum = Float64.(k_sum) #/ nT
        if verbose
            return k_sum, k_evals, khat_evals, dval, Qneghalf
        else
            return k_sum
        end
    end

    return k_faster
end
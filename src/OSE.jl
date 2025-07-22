export 
    symDist,
    normWPieces,
    makeRbfKer,
    makebwKer,
    makeNormKernel_rbf,
    makeNormKernel_cone,
    evalPhi

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

function normWPieces(X)
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

function makebwKer(bw)
    # assemble initial kernel
    function k(x, y, xpre, ypre)
        D = euclidean(x, y)
        σ = ConsistentKoopman.conebw(x, y, xpre, ypre).^-1
        return exp.(-1 * D.^2 ./ (σ .^ 2 * bw .^ 2))
    end
    return k
end

function makeNormKernel_rbf(W, X, nT, bw)
    Qneghalf, K̂ = normWPieces(W)
    
    k = makeRbfKer(bw)

    function k_faster(y; verbose = false)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] = k(y, X[:, j])
        end
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) .* Qneghalf
        
        k_sum = zeros(nT)
        Threads.@threads for l = 1:nT
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

function makeNormKernel_cone(W, X, nT, bw)
    Qneghalf, K̂ = normWPieces(W)
    
    k = makebwKer(bw)

    function k_faster(y, ypre; verbose = false)

        k_evals = zeros(nT)
        Threads.@threads for j = 1:nT
            k_evals[j] = k(y, X[:, j + 1], ypre, X[:, j])
        end
       
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) .* Qneghalf
        k_sum = zeros(nT)
        Threads.@threads for l = 1:nT
            if (l % 100 == 0) & verbose
                println(l)
            end
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

function makeNormKernel_cone(W, X, nT, bw, NN)
    if NN == 0
        return makeNormKernel_cone(W, X, nT, bw)
    end

    Qneghalf, K̂ = normWPieces(W)
    
    k = makebwKer(bw)

    function k_faster(y, ypre; verbose = false, verbose2 = true)

        k_evals = zeros(nT)
        d_evals = zeros(nT)

        Threads.@threads for j = 1:nT
            if (j % 200 == 0) & verbose2
                println(j)
            end
            d_evals[j] = euclidean(y, X[:, j + 1])
            k_evals[j] = k(y, X[:, j + 1], ypre, X[:, j])
        end

        k_sort = sortperm(d_evals)

        if verbose
            println("length eliminated: ", length(k_sort[NN + 1:end]))
            println("max elim: ", maximum(k_evals[k_sort[NN + 1:end]]))
            println("min elim: ", minimum(k_evals[k_sort[NN + 1:end]]))
        end

        k_evals[k_sort[(NN + 1):end]] .= 0
       
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) .* Qneghalf
        k_sum = zeros(nT)
        Threads.@threads for l = 1:nT
            if (l % 100 == 0) & verbose2
                println(l)
            end
            for j = 1:nT
                k_sum[l] += khat_evals[j] *  K̂[l, j] 
            end
        end

        k_sum = Float64.(k_sum) #/ nT
        if verbose
            return k_sum, k_evals, khat_evals, dval, Qneghalf, d_evals
        else
            return k_sum
        end
    end

    return k_faster
end

function evalPhi(kernevals::AbstractVector, φ::AbstractVector, κ::Number)
    return sum(kernevals .* φ) / κ
end
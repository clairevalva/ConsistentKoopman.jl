# do NLSA part only for MJO for compatibility with Nystrom extension
using DoubleFloats
# X is any data
# NN = 0, for now I gueess 

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


function normW_quad(X)
    X = Double64.(X)
    D = sum(X, dims = 2)[:] 
    Dinv = Diagonal(D.^(-1))
    S = sum(X ./ (D'), dims = 2)[:]
    Sneghalf = Diagonal(S.^(-1/2))

    K̂ = Dinv * X * Sneghalf
    K̃ = K̂ * (K̂')

    return K̃
end

function normWPieces(X::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}})
    # AHHH
    X = Double64.(X)
    D = sum(X, dims = 2)[:]
    Dinv = Diagonal(D.^(-1))
    S = sum(X ./ (D'), dims = 2)[:]
    Sneghalf = Diagonal(S.^(-1/2))

    K̂ = Dinv * X * Sneghalf
    

    return diag(Sneghalf), K̂
end

function normWPieces_single(X::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}})
    # AHHH
    X = X
    D = sum(X, dims = 2)[:]
    Dinv = Diagonal(D.^(-1))
    S = sum(X ./ (D'), dims = 2)[:]
    Sneghalf = Diagonal(S.^(-1/2))

    K̂ = Dinv * X * Sneghalf
    

    return diag(Sneghalf), K̂
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



function makeNormKernel_cone(W, X, nT, bw)
    Qneghalf, K̂ = normWPieces(W)
    
    k = makebwKer(bw)

    println(nT)
    function k_faster(y, ypre; verbose = false)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] = k(y, X[:, j + 1], ypre, X[:, j])
        end
        k_evals = Double64.(k_evals)
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) 
        println("here")
        k_sum = zeros(Double64, nT)
        for l = 1:nT
            jvals = zeros(Double64, nT)
            for j = 1:nT
                # k_sum[l] += khat_evals[j] *  K̂[j, l] * Qneghalf[j] # replace this naive summing alg with something better
                jvals[j] = khat_evals[j] *  K̂[j, l] * Qneghalf[j]
            end
            k_sum[l] = sum(jvals)
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


function makeNormKernel_cone_single(W, X, nT, bw, NN::Integer)
    # consider use of sparse matrices here? would make everything simpler

    if NN == 0
        return makeNormKernel_cone_single(W, X, nT, bw)
    end

    Qneghalf, K̂ = normWPieces_single(W)
    
    k = makebwKer(bw)


    function k_faster(y, ypre; verbose = false)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] = k(y, X[:, j + 1], ypre, X[:, j])
        end
        k_evals = k_evals

        k_sort = sortperm(-1*k_evals)
        println("length eliminated: ", length(k_sort[NN + 1:end]))
        println("max elim: ", maximum(k_evals[k_sort[NN + 1:end]]))
        println("min elim", minimum(k_evals[k_sort[NN + 1:end]]))

        k_evals[k_sort[NN + 1:end]] .= 0
        
        dval = sum(k_evals) # / nT

        khat_evals = k_evals * (dval ^-1) 
        println("here")
        k_sum = zeros(nT)
        for l = 1:nT
            jvals = zeros(nT)
            for j = 1:nT
                # k_sum[l] += khat_evals[j] *  K̂[j, l] * Qneghalf[j] # replace this naive summing alg with something better
                # jvals[j] = khat_evals[j] *  K̂[j, l] * Qneghalf[j]
                if k_evals[j] == 0
                    jvals[j] = 0
                else
                    jvals[j] = khat_evals[j] *  K̂[j, l] * Qneghalf[j]
                end
            end
            k_sum[l] = sum(jvals)
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



function makeNormKernel_cone_single(W, X, nT, bw)
    Qneghalf, K̂ = normWPieces_single(W)
    
    k = makebwKer(bw)


    function k_faster(y, ypre; verbose = false)
        k_evals = zeros(nT)

        for j = 1:nT
            k_evals[j] = k(y, X[:, j + 1], ypre, X[:, j])
        end
        k_evals = k_evals
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) 
        println("here")
        k_sum = zeros(nT)
        for l = 1:nT
            jvals = zeros(nT)
            for j = 1:nT
                # k_sum[l] += khat_evals[j] *  K̂[j, l] * Qneghalf[j] # replace this naive summing alg with something better
                jvals[j] = khat_evals[j] *  K̂[j, l] * Qneghalf[j]
            end
            k_sum[l] = sum(jvals)
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
        k_evals = Double64.(k_evals)
        dval = sum(k_evals) # / nT
        
        khat_evals = k_evals * (dval ^-1) .* Qneghalf
        println("here")
        k_sum = zeros(Double64, nT)
        # for l = 1:nT
        #     jvals = zeros(Double64, nT)
        #     for j = 1:nT
        #         # k_sum[l] += khat_evals[j] *  K̂[j, l] * Qneghalf[j] # replace this naive summing alg with something better
        #         jvals[j] = khat_evals[j] *  K̂[j, l] * Qneghalf[j]
        #         jvals[j] = khat_evals[j] *  K̂[j, l] #* Qneghalf[j]
        #     end
        #     k_sum[l] = sum(jvals)
        # end
        println("here 2")
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
# function makeNormKernel_cone(W, X, nT, bw, NN::Integer)
#     if NN == 0
#         println("use no NN version of function")
#         k_fun = makeNormKernel_cone(W, X, nT, bw)
#         return k_fun
#     end

#     Qneghalf, K̂ = normWPieces(W)
    
#     # assemble initial kernel
#     k = makebwKer(bw)


#     function k_faster(y, ypre)
#         k_evals = zeros(nT)

#         for j = 1:nT
#             k_evals[j] += k(y, X[:, j + 1], ypre, X[:, j])
#         end

#         # just zero out the small evals?
#         k_sort = sortperm(k_evals)
#         k_evals[k_sort[NN + 1:end]] .= 0

#         dval = sum(k_evals) # / NN
        
#         khat_evals = (k_evals ./ dval) .* Qneghalf

#         k_sum = zeros(nT)
#         for l = 1:nT
#             for j = 1:nT
#                 k_sum[l] += khat_evals[j] *  K̂[j, l]
#             end
#         end

#         k_sum = k_sum #/ NN

#         return k_sum
#     end

#     return k_faster
# end


# function makePsis_cone(phis, lams,nDiff,nT, k_norm::Function)

#     function psis(y, ypre)
#         k_evals = k_norm(y, ypre)
#         psi = zeros(nDiff)

#         for j = 1:nDiff
#             psi[j] = (nT * sqrt(lams[j]))^(-1) * sum(k_evals .* phis[:, j])
#         end

#         # I kinda think NNs should be included? if its implicit here?
        
#         return psi
#     end
# end

# function makePsis_cone(phis, lams,nDiff,nT, k_norm::Function, NN::Integer)

#     function psis(y, ypre)
#         k_evals = k_norm(y, ypre)
#         psi = zeros(nDiff)

#         for j = 1:nDiff
#             psi[j] = (NN * sqrt(lams[j]))^(-1) * sum(k_evals .* phis[:, j])
#         end

#         # I kinda think NNs should be included? if its implicit here?
        
#         return psi
#     end
# end


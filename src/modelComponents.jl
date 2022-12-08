"""
    tune bandwidth function -> should make sure this works, the behavior is a lil funky
        fixed, is a small typo in the pseudo code I think (is missing a square?)

    from 10.1016/j.acha.2017.09.001
    probably cheating a bit by only taking the sum of the NN nearest neighbors
    hoping it doesn't matter too much 

"""

function tune_bandwidth(D::Matrix{Float64}, DN::Matrix{Integer}, 
    NN::Integer, nT::Integer, epsls::Vector{Float64})
    
    Dsq = D.^2
    ris = sum(Dsq, dims = 2) ./ (NN - 1)

    nE = length(epsls)
    eps_sum = zeros(Float64, nE)
    
    Kel_sum = 0 
    for eee in 1:nE
        eps = epsls[eee]
        for m = 1:nT
            for n = 1:NN
                k = DN[m,n]
                Kel_sum += exp(-1*Dsq[m,n] / (eps * ris[m] * ris[k])^2)
                
            end
        end
        eps_sum[eee] = Kel_sum
    end

    eps_sum_l = log.(eps_sum)
    epsls_l = log.(epsls)

    eps_sum_prime = (eps_sum_l[2:end] - eps_sum_l[1:(end - 1)]) ./ (epsls_l[2:end] - epsls_l[1:(end - 1)])
    best_eps = epsls[argmax(eps_sum_prime)]

    return best_eps, eps_sum_prime
end


function sparseW_mb(X::Matrix{Float64}, eps::Float64;
    NN::Integer = 0, usenorm::Function = norm, sym::Bool = true )
   # get distances
   _, N = distNN(X[1:(end - 1),:], NN, usenorm = usenorm)
   
   nT = size(X, 1) - 1
   if NN == 0
       NN = nT
   end

   W = zeros(Float64, nT, nT)

   for i = 1:nT
       for j = 1:NN
           k = N[i,j]
           if i != k
                W[i, k] = varbandwidth_kernel(X[i + 1,:], X[k + 1,:], X[i,:], X[k,:], γ = eps)
           end
       end
   end

   if sym
       W = sym_M(W)
   end

   return W
end
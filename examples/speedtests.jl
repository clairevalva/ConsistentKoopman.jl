using Distances
using BenchmarkTools

# make some data
include("torusData.jl") 

function make_distances_1(X, nT)
    D = zeros(Float64, nT, nT)
    
    # if no nearest neighbors specified, keep all of them
    for i = 1:nT
        for j = 1:(i-1)
            D[j,i] = euclidean(X[:,j], X[:,i])
        end
    end
    D = D + D'
    return D
end

function make_distances_2(X)
    R = pairwise(euclidean, X, dims=2)
    return R
end


x_test = x[:, 1:500]
_, nT = size(x_test)
test1 = make_distances_1(x_test, nT)
test2 = make_distances_2(x_test)
mean(test1 - test2)

time1 = @benchmark make_distances_1(x_test, nT)
# 3.571 ms (249504 allocations: 26.66 MiB)

time2 = @benchmark make_distances_2(x_test)
# 3.571 ms (249504 allocations: 26.66 MiB)

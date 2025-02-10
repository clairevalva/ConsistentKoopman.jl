# from variational Koopman initial repo
using ForwardDiff

"""
    makeIO(k, X::AbstractVector)

    makes integral operator from function k, evaluated at points X, returns K, s.t.
        Kv(x) = ∑ k(x, y) * v(y) / N = ∫ k(x, y) v(y) dμₘ(y)
    
    several versions of function to take care of faster cases, 
        here X can be vector valued (i.e. vector of static vectors or when scalar measurements), but V is identically 1

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[k]
    - X: vector 
"""
function makeIO(k, X::AbstractVector)
    N = size(X, 1)
    function g(y)
        gy = 0.0
        for j = 1:N
            gy += k(y, X[j])
        end

        return gy / N
    end
end

"""
    makeIO(k, X)

    makes integral operator from function k, evaluated at points X, returns K, s.t.
        Kv(x) = ∑ k(x, y) * v(y) / N = ∫ k(x, y) v(y) dμₘ(y)
    
    several versions of function to take care of faster cases, 
        here X can matrix valued (nD × nT ), but V is identically 1

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[:, k]
    - X: matrix of measurements (nD × nT )
"""
function makeIO(k, X)
    N = size(X, 2)
    function g(y)
        gy = 0.0
        for j = 1:N
            gy += k(y, X[:, j])
        end
        return gy / N
    end
end


"""
    makeIO(k, X::AbstractVector)

    makes integral operator from function k, evaluated at points X, returns K, s.t.
        Kv(x) = ∑ k(x, y) * v(y) / N = ∫ k(x, y) v(y) dμₘ(y)
    
    several versions of function to take care of faster cases, 
        here X can be vector valued (i.e. vector of static vectors or when scalar measurements)

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[k]
    - X: vector 
    - v: values of function v at measured x
"""
function makeIO(k, X::AbstractVector, V)
    N = size(X, 1)
    function g(y)
        gy = 0.0
        for j = 1:N
            gy += k(y, X[j]) * V[j]
        end

        return gy / N
    end
end

"""
    makeIO(k, X, V)

    makes integral operator from function k, evaluated at points X, returns K, s.t.
        Kv(x) = ∑ k(x, y) * v(y) / N = ∫ k(x, y) v(y) dμₘ(y)
    
    several versions of function to take care of faster cases, 
        here X can matrix valued (nD × nT )

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[:, k]
    - X: matrix of measurements (nD × nT )
    - v: values of function v at measured x
"""
function makeIO(k, X, V)
    N = size(X, 2)
    function g(y)
        gy = 0.0
        for j = 1:N
            gy += k(y, X[:, j]) * V[j]
        end

        return gy / N
    end
end



"""
    inclusion(g, X::AbstractVector)
    
    inclusion operator for function g on X

    Arguments
    =================
    - k: funciton, takes arguments of the form of X[k]
    - X: vector 
"""
function inclusion(g, X::AbstractVector)
    N = size(X, 1)
    v = g.(X)
    return v

end

"""
    inclusion(g, X)
    
    inclusion operator for function g on X

    Arguments
    =================
    - k: funciton, takes arguments arguments of the form of X[:, k]
    - X: vector 
"""
function inclusion(g, X)
    N = size(X, 2)
    v = ones(N)
    for j = 1:N
        v[j] = g(X[:,j])
    end
    return v

end


"""
    composek(k1, k2, X::AbstractVector)
    
    composes two kernels k with the sampling of X

    Arguments
    =================
    - k1: kernel function, takes two arguments of the form of X[k]
    - k2: kernel function, takes two arguments of the form of X[k]
    - X: vector 
"""
function composek(k1, k2, X::AbstractVector)
    N = size(X, 1)
    function w(x,y)
        w = 0.0
        for j = 1:N
            w += k1(x, X[j]) * k2(X[j], y)
        end
        return w / N
    end

    return w

end

"""
    composek(k1, k2, X)
    
    composes two kernels k with the sampling of X

    Arguments
    =================
    - k1: kernel function, takes two arguments of the form of X[:, k]
    - k2: kernel function, takes two arguments of the form of X[:, k]
    - X: vector 
"""
function composek(k1, k2, X)
    N = size(X, 2)
    function w(x,y)
        w = 0.0
        for j = 1:N
            w += k1(x, X[:,j]) * k2(X[:,j], y)
        end
        return w / N
    end

    return w

end


"""
    swap_args(g)
    
    swaps arguments of function g (for kernels, useless in symmetric cases)

    Arguments
    =================
    - g: function, takes two arguments
"""
function swap_args(g)
    h(y, x) = g(x, y)
    return h
end

### nonsymmetric bistochastic normalization ###
"""
    bs_norm(k, X)
    
    implements nonsymmetric bistochastic normalization of a kernel

    Arguments
    =================
    - k: function, takes two arguments
    - X: vector 
"""
function bs_norm(k, X)
    d = makeIO(k, X)
    k_r(x,y) = k(x,y) / d(y)
    q = makeIO(k_r, X)
    k_q(x,y) = k(x,y) / sqrt(q(y))
    k_bs(x, y) = k_q(x,y) / d(x)

    return k_bs
end

"""
    bssym_norm(k, X)
    
    implements symmetric bistochastic normalization of a kernel
    is EXTREMELY SLOW, probably better to multiply the linear operators

    Arguments
    =================
    - k: function, takes two arguments
    - X: vector 
"""
function bssym_norm(k, X)
    k_bs = bs_norm(k, X)

    k_sym = composek(k_bs, swap_args(k_bs), X)
    return k_sym
end

"""
    bssym_norm(k, X)
    
    implements symmetric squareroot norm of a kernel

    Arguments
    =================
    - k: function, takes two arguments
    - X: vector 
"""
function sqrtsym_norm(k, X)
    q = makeIO(k, X)

    k_sym(x,y) = k(x, y) / sqrt(q(x) * q(y))
    return k_sym
end


"""
    make_vk(k, v)
    
    return V_1k for kernel k (i.e. directional derivative of the first argument of a function)

    Arguments
    =================
    - k: function, takes two arguments
    - X: v, function that computes derivative
"""
function make_vk(k, v)
    # get directional derivative of first argument
    function vk(x, y)
        kpartial(x) = k(x, y)
        # jac_k = ForwardDiff.gradient(kpartial, x)' * v(x)
        jac_k = ForwardDiff.derivative(kpartial, x)' * v(x)
        return jac_k
    end
end


"""
    make_vk(k, v)
    
    return V_2k for kernel k (i.e. directional derivative of the second argument of a function)

    Arguments
    =================
    - k: function, takes two arguments
    - v: function that computes derivative
"""
function make_kv(k,v)
    # get directional derivative of second argument
    function vk(x, y)
        kpartial(y) = k(x, y)
         # check that this works for more complicated examples
        # jac_k = ForwardDiff.gradient(kpartial, y)' * v(y)
        jac_k = ForwardDiff.derivative(kpartial, y)' * v(y)
        return jac_k
    end
end


"""
    make_rhs_symmk(k, v, z::Real, X)
    
    makes the rhs of eigenvalue problem for symmetric kernel (should return K(z^2 + V^2)K^*)

    Arguments
    =================
    - k: function, takes two arguments
    - v: function that computes derivative
    - z: scalar parameter
    - X: vector of measurements
"""
function make_rhs_symmk(k, v, z::Real, X)
    ### should return K(z^2 + V^2)K^*
    
    k_k = composek(k, k, X)
    kv = make_kv(k, v)
    vk = make_vk(k, v)
    kv_vk = composek(kv, vk, X)
    k_rhs(x, y) = z * z * k_k(x, y) - kv_vk(x,y)
        

    return k_rhs
end

"""
    make_rhs_nosymmk(k, v, z::Real, X)
    
    makes the rhs of eigenvalue problem for nonsymmetric kernel (should return K(z^2 + V^2)K^*)

    Arguments
    =================
    - k: function, takes two arguments
    - v: function that computes derivative
    - z: scalar parameter
    - X: vector of measurements
"""
function make_rhs_nosymmk(k, v, z::Real, X)
    ### should return K(z^2 + V^2)K^*
    
    kt = swap_args(k)
    k_kt = composek(k, kt, X)
    kv = make_kv(k, v)
    vkt = make_vk(kt, v)
    kv_vkt = composek(kv, vkt, X)
    k_rhs(x, y) = z * z * k_kt(x, y) - kv_vkt(x,y)

    return k_rhs
end

"""
    make_rhs(k, v, z::Real, X, symmetric = true)
    
    makes the rhs of eigenvalue problem, pass symmetric arg,
    should return K(z^2 + V^2)K^*

    Arguments
    =================
    - k: function, takes two arguments
    - v: function that computes derivative
    - z: scalar parameter
    - X: vector of measurements
    - symmetric: bool to determine if k is symmetric
"""
function make_rhs(k, v, z::Real, X; symmetric = true)
    if symmetric
        return make_rhs_symmk(k, v, z, X)
    else
        return make_rhs_nosymmk(k, v, z, X)
    end
end

"""
    make_lhs(k, v, X, symmetric = true)
    
    makes the lhs of eigenvalue problem, pass symmetric arg,
    should return KK^*KVK^*KK^* where we average KVK^* and (KVK^*)^*

    Arguments
    =================
    - k: function, takes two arguments
    - v: function that computes derivative
    - X: vector of measurements
    - symmetric: bool to determine if k is symmetric
"""
function make_lhs(k, v, X; symmetric = true)
    ### should return KK^*KVK^*KK^* where we average KVK^* and (KVK^*)^*
    if symmetric

        vk = make_vk(k, v)
        k_vk = composek(k, vk, X)
        k_vk_t = swap_args(k_vk)

        k_vk_avg(x, y) = 0.5*(k_vk(x, y) - k_vk_t(x, y))

        k_lhs = composek(k, composek(k_vk_avg, k, X), X)
        return k_lhs
    else
        kt = swap_args(k)
        vkt = make_vk(kt, v)
        k_vkt = composek(k, vkt, X)
        k_vkt_t = swap_args(k_vkt)

        k_vkt_avg(x, y) = 0.5*(k_vkt(x, y) - k_vkt_t(x, y))

        k_lhs = composek(k, composek(k_vkt_avg, kt, X), X)
        
        return k_lhs
    end
    
end
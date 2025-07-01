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
    make_Sϵ(X::AbstractArray, k)

    computes function S(ϵ) where S(ϵ) = ∫ k(x, x, param = ϵ) dμ(x)

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[k]
    - X: matrix of measurements (nD × nT )
"""
function make_Sϵ(X::AbstractVector, k)
    N = size(X, 1)
    function estSϵ(ϵ)
        S = 0.0
        for j = 1:N
            for i = 1:N
                S += k(X[j], X[i], exp(ϵ))
            end
        end

        return log(S)
    end
end

"""
    make_Sϵ(X, k)

    computes function S(ϵ) where S(ϵ) = ∫ k(x, x, param = exp(ϵ)) dμ(x)

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[:, k]
    - X: matrix of measurements (nD × nT )
"""
function make_Sϵ(X, k)
    N = size(X, 2)
    function estSϵ(ϵ)
        S = 0.0
        for j = 1:N
            for i = 1:N
                S += k(X[:,j], X[:,i], exp(ϵ))
            end
        end

        return log(S)
    end
end


"""
    tunek(X, k, testparams)

    finds best ϵ based on maximization procedure
    TODO: pull citations (Berry and Harlim 2016, Berry et al 2015)

    note that bestϵ = exp.(testparams[best_epsj]) can be seen as an approximate dimension of the manifold M

    Arguments
    =================
    - k: kernel function, takes two arguments of the form of X[:, k]
    - X: matrix of measurements (nD × nT )
    - testparams: vector of possible ϵ (to be put into kernel as exp(ϵ))

"""
function tunek(X, k, testparams)
    S = make_Sϵ(X, k)
    S_grad(ϵ) = ForwardDiff.derivative(S, ϵ)

    Sgs = S_grad.(testparams)
    best_epsj = argmax(Sgs)
    bestϵ = exp.(testparams[best_epsj])

    # bestϵ

    return best_epsj, bestϵ, Sgs[best_epsj]
end


""" 
    estvolM(k, X, dimM, ϵ)

    estimates volume of manifold M 

   Arguments
    =================
    - k: function, takes two arguments
    - X: vector 
    - dimM: bestϵ from tunek can be seen as an approximate dimension of the manifold M
    - ϵ: bandwidth choice (choose dimM = ϵ) I think


"""

function estvolM(k, X, dimM, ϵ)
    N = size(X, 1)
    d = makeIO(k, X)

    p(x, y) = k(x, y) / d(x)

    # "integrate p(x, x)"
    intp = 0.0
    for j = 1:N
        intp += p(X[j], X[j])
    end
    intp /= N

    return (π * ϵ^2)^(dimM / 2) * intp
end


"""
    indbwcts(k, X)
    
    get individual bandwidth function

    Arguments
    =================
    - k: function, takes two arguments
    - X: vector 
    - volM: extimate of the volume of the manifold
    - dimM: bestϵ from tunek can be seen as an approximate dimension of the manifold M
"""
function indbwcts(k, X::AbstractVector, volM::Real, dimM)
    N = size(X, 1)
    d = makeIO(k, X)

    # "integrate d" compute approx volume
    intd = 0.0
    for j = 1:N
        intd += d(X[j])
    end
    intd /= N

    ρ(x) = intd / (volM * d(x))
    r(x) = ρ(x)^(-dimM^-1)

    return r
end

"""

    indbwdis(X::AbstractVector, dimM)

    makes function to get individual bandwidths, as in discrete/matrix formulation

"""

function indbwdis(X::AbstractVector, usenorm = euclidean)

    NN = size(X, 1)
    # make sure x has size (1, nD)

    function σ(x)
        
        normdiff = 0
        for j = 1:NN
            normdiff += euclidean(X[j], x)^2
        end

        
        return sqrt(normdiff/ (NN - 1))
    end

    return σ
end


"""

    indbwdis(X::Matrix, dimM)

    makes function to get individual bandwidths, as in discrete/matrix formulation

"""

function indbwdis(X)

    NN = size(X, 2)
    # make sure x has size (1, nD)

    function σ(x)
        
        normdiff = 0
        for j = 1:NN
            normdiff += euclidean(X[:, j], x)^2
        end

        
        return sqrt(normdiff/ (NN - 1))
    end

    return σ
end

"""

    makesepbwk(X, γ::Real, dimM::Real, usenorm::Function = norm)

    makes seperable bandwidth kernel using same idea matrix (but for continuous kernels)
    probably faster to first tune in discrete space

    TO DO: test!

"""
function makesepbwk(X, dimM::Real, usenorm = norm)
    σ = indbwdis(X)

    function k_sb(x,y, γ)
        bw = (σ(x) * σ(y)) ^ (1 / dimM)
        return exp(-1*usenorm(x, y) / (γ^2 * bw))
    end

    return k_sb
end





export 
    centeredDiff,
    makeVf,
    makeΔ,
    diffV,
    doKoopman_diff


# functions built to replicate method https://doi.org/10.1038/s41467-021-26357-x

"""
    centeredDiff(f::AbstractVector, dt::Real)

    derivative approximation for function (vector f) using a centered difference scheme

    Arguments
    =================
    - f: discretization of vector
    - dt: sampling distance in time

    tested with the following on Nov 21, 2024

    dt = 0.1
    x = 0:dt:(2*pi + dt)
    f = sin.(x)
    fprime = cos.(x)
    fapprox = centeredDiff(f, dt)

"""

function centeredDiff(f::AbstractVector, dt::Real)

    Vf = 1/12*(circshift(f, 2) - circshift(f, -2)) +  2/3*(circshift(f, -1) - circshift(f, 1))
    Vf[1:2] .= 0
    Vf[(end - 1):end] .= 0

    return Vf/dt
end

"""
    makeVf(fs::AbstractMatrix, dt::Real)

    make symmetric approximation of V using centered differences on functions f (matrix f)

    Arguments
    =================
    - fs: matrix of size nT × nFs (i.e. time by num of functions)
    - dt: sampling distance in time

    tested with the following on Nov 21, 2024
    dt = 0.1
    Vf_unsym = makeVf(φ, dt)
    Vf_unsym[:, 5] == centeredDiff(φ[:, 5], dt)

""" 

function makeVf(fs::AbstractMatrix, dt::Real)
    Vf_unsym = mapslices(x -> centeredDiff(x, dt), fs, dims = 1)
    N = size(fs, 1)
    Ṽ = (fs' * Vf_unsym) / N
    Ṽ_sym = (Ṽ - Ṽ')/2
    
    return Ṽ_sym 
end

"""
    makeΔ(κs::AbstractVector)

    makes diffusion operatir

    Arguments
    =================
    - κs: vector of NLSA eigenvalues

"""

function makeΔ(κs::AbstractVector)
    E = 1 ./κs .- 1
    E ./= E[2]
    return Diagonal(E)
end

"""
    diffV(V::AbstractMatrix, Δ::AbstractMatrix, ϵ::Real)

    makes diffusion approximation of V

    Arguments
    =================
    - V: approximate generator (symmetric)
    - Δ: diffusion operator, same size as V, built from NLSA
    - ϵ: diffusion parameter

"""

function diffV(V::AbstractMatrix, Δ::AbstractMatrix, ϵ::Real)
    return V - ϵ*Δ
end

"""
    doKoopman_diff(fs::AbstractMatrix, κs::AbstractVector, dt::Real, ϵ::Real)

    approximate Koopman operator as in https://doi.org/10.1038/s41467-021-26357-x

    Arguments
    =================
    - fs: size nT × nEig (time by functions), nlsa eigenfunctions
    - κs: size nEig, corresponding NLSA eigenvalues
    - dt: sampling frequency for nlsa eigenfunctions
    - ϵ: diffusion parameter

"""

function doKoopman_diff(fs::AbstractMatrix, κs::AbstractVector, dt::Real, ϵ::Real)
    V = makeVf(fs, dt)
    Δ = makeΔ(κs)
    W = diffV(V, Δ, ϵ)

    λ, u = eigen(W, sortby = x -> -1*real(x))
    nu = λ / (2*π)
    gs = 1im * zeros(size(fs))
    L = size(fs, 2)

    Threads.@threads for j = 1:L
        for i = 1:L
            gs[:, j] .+= u[i, j] * fs[:, i]
        end
    end

    return λ, u, nu, gs
end


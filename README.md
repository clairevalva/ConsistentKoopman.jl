# Consistent Koopman
Julia implementations for consistent (spectral) approximations of Koopman operators, following [arXiv:2309.00732](https://doi.org/10.48550/arXiv.2309.00732) that includes a _very_ limited implementation of nonlinear Laplacian spectral analysis or NLSA and related kernel algorithms. (The full and original package — implemented in Matlab — can be found here: [NLSA](https://github.com/dg227/NLSA).)

See the examples folder [and flattorus.jl](https://github.com/clairevalva/ConsistentKoopman.jl/blob/main/examples/flattorus.jl) for the computation of approximate Koopman eigenfunctions of the flat torus system:
\[ \Phi^t: \mathbf{T}^2 \to \mathbf{T}^2, \: \Phi^t(\theta_1, \theta_2) = (\theta_1 + \alpha_1 t, \theta_2 + \alpha_2 t) \mathrm{mod} 2\pi \]
for which eigenfunctions should resemble foruier modes and the eigenvalues should be integer linear combinations of $\alpha_1$ and $\alpha_2$.




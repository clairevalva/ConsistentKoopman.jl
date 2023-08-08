# compare values from matlab package
fid = h5open("/Users/clairevalva/codez/NLSA/examples/diffvals_short_r3.h5", "r")

origλ = read(fid["lam2"])
origμ = read(fid["mu2"])
origφ = read(fid["phi2"])

plot(1:407,real(normφ[1:407,2]))
plot!(1:407, real(origφ[1:407,2]))
# for some reason, this appears much closer to the R4 eigenfunctions?

# old μ or w is 0.2441
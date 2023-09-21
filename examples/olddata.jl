# TODO test current
# TODO projection to physical space (and reconstruction?)
include("torusData.jl")
include("../src/kernels.jl")
include("../src/utils.jl")
include("../src/modelComponents.jl")
include("../src/resolvent.jl")
using Plots
using Statistics
using HDF5
using StatsBase

κ_old = h5open("/Users/clairevalva/codez/NLSA/lamb_shortr4.mat", "r")["lambda"]
κ_old = κ_old[:,1]
φ_old = h5open("/Users/clairevalva/codez/NLSA/phi_shortr4.mat", "r")["phi"]
φ_old = φ_old[:,:]
R_z = h5open("/Users/clairevalva/codez/NLSA/R_z.mat", "r")["R_z"][:,:]
others = h5open("/Users/clairevalva/codez/NLSA/examples.mat", "r")
U0 = h5open("/Users/clairevalva/codez/NLSA/U0.mat", "r")["U0"][:,:]

U = others["U"][:,:]
P = others["P"][:,:]
c = others["c"][:,:]
frequencies = others["frequencies"][:,:]
G_old = others["G"][:,:]

newX = h5open("/Users/clairevalva/codez/NLSA/x_orig.mat", "r")["x"][:,:]

R_z_unw = 1im*zeros(101, 101)
U_uw = 1im*zeros(101, 101)
P_uw = 1im*zeros(101, 101)
c_uw = 1im*zeros(101, 101)
U0_uw = 1im*zeros(101, 101)
for i in 1:101
    for j in 1:101
        R_z_unw[i,j] = R_z[i,j].real + 1im*R_z[i,j].imag
        P_uw[i,j] = P[i,j].real + 1im*P[i,j].imag
        U_uw[i,j] = U[i,j].real + 1im*U[i,j].imag
        c_uw[i,j] = c[i,j].real + 1im*c[i,j].imag
        U0_uw[i,j] = U0[i,j].real + 1im*U0[i,j].imag
    end
end



U_test, P_test = polar(R_z_unw)
maximum(abs.(U_test - U_uw))
maximum(abs.(P_test - P_uw))

φ_plus = posfilter(φ_old)
U0_test = koopmanop([1], φ_plus, [1 / 4096])[1,1:101, 1:101]
maximum(abs.(U0_test - U0_uw))

Rz_test = resolventop_power(φ_plus, [1 / 4096], 50, dt, 1.)
maximum(abs.(Rz_test[1:101, 1:101] - R_z_unw))
U, P = polar(Rz_test[1:101, 1:101])
Ptau_test = G * P * G
F_test = eigen(Ptau_test)
gamma = polarfz(real(F_test.values), 1.)
frequencies_test = imag(1 ./gamma)


G_test = Gtau(κ_old, 1e-4)
maximum(abs.(G - G_test[1:101, 1:101]))


zeta = others["zeta"][:,:]
zeta_uw = 1im*zeros(4096, 101)
for i in 1:4096
    for j in 1:101
        zeta_uw[i,j] = zeta[i,j].real + 1im*zeta[i,j].imag
    end
end

# want the below
using Plots
# params of system
f = 30^0.5 # frequency along theta coordinate
φ_amp = 1
θ_amp = 1
φ_rad = 0.5
θ_rad = 0.5 

nS = 0 # number of spin up samples
T = 200

# params of wanted sample
nST = 128;
nS = nST * T;
dt = 2 * pi / nST; #0.05

nSpin = 0
# params of sample format (R3 or R4)
SF = "R3"

ts = 0:dt:(nS*dt)

# do the time series
if φ_amp == 1
    φ = ts
else
    φ = 2*atan( 1 + sqrt(1 - φ_amp)) * tan.(sqrt(φ_amp) .* ts / 2 ) ./ sqrt(φ_amp)
    
end


if θ_amp == 1
    θ = f .* ts .+ (pi / 2);
else
    θ = 2*acot(sqrt(1 - θ_amp)) .+ sqrt.(θ_amp)*cot.(sqrt(θ_amp) * f .* ts ./ 2 )
end

φ = φ .% 2pi
θ = θ .% 2pi

ts = ts[nSpin + 1:end]
φ = φ[nSpin + 1:end]
θ = θ[nSpin + 1:end]

nT = size(ts, 1)

if SF == "R3"
    x = zeros(3, nT)
    x[1,:] = (1 .+ φ_rad * cos.(θ)) .* cos.(φ)
    x[2,:] = (1 .+ φ_rad * cos.(θ)) .* sin.(φ)
    x[3,:] = θ_rad * sin.(θ)
elseif SF == "R4"
    x = zeros(4,nT)
    x[1,:] = cos.(φ)
    x[2,:] = sin.(φ)
    x[3,:] = cos.(θ)
    x[4,:] = sin.(θ)
else
    println("unimplemented sample format, try 'R3' or 'R4' ")
end

# plot(ts, x')
# xlims!(1,2000)
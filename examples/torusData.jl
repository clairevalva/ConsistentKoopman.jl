# want the below
# sorta tested on Nov 4, 2022 by comparing to orig version for 1 case.
# % dphi/dt   =       1 + sqrt( 1 - aPhi ) * cos( phi ) 
# % dtheta/dt = f * ( 1 - sqrt( 1 - aTheta ) * sin( theta ) )
# %
# % phi( 0 )   = 0
# % theta( 0 ) = 0

# params of system
f = 30^0.5 # frequency along theta coordinate
φ_amp = 1
θ_amp = 1
φ_rad = 0.5
θ_rad = 0.5 

# params of wanted sample
dt = 0.05
nS = 20 # number of spin up samples
T = 320

# params of sample format (R3 or R4)
SF = "R3"

ts = 0:dt:(T + nS*dt)

# do the time series
if φ_amp == 1
    φ = ts
else
    φ = 2*atan( 1 + sqrt(1 - φ_amp)) * tan.(sqrt(φ_amp) .* ts / 2 ) ./ sqrt(φ_amp)
end

if θ_amp == 1
    θ = f .* ts .+ pi ./ 2;
else
    θ = 2*acot(sqrt(1 - θ_amp)) .+ sqrt.(θ_amp)*cot.(sqrt(θ_amp) * f .* ts ./ 2 )
end

ts = ts[nS:end]
φ = φ[nS:end]
θ = θ[nS:end]

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
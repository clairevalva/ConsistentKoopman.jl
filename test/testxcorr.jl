using LinearAlgebra
using Plots
using StatsBase



function xcorr(x::AbstractVector, y::AbstractVector, lag::Int; coeff = "innerprod")
    rawcorr = sum(x[(lag + 1):end] .* conj.(y[1:(end - lag)]))

    if coeff == "innerprod"
        xnorm = sum(x[(lag + 1):end] .* conj.(x[(lag + 1):end]))
        ynorm = sum(y[1:(end - lag)] .* conj.(y[1:(end - lag)]))

        usec = sqrt(xnorm * ynorm)
        # usec = sqrt((x[(lag + 1):end]' * conj.(x[(lag + 1):end])) * (y[1:(end - lag)]' * conj.(y[1:(end - lag)])))
    elseif coeff == "normalized"
        xnorm = sum(x .* conj.(x))
        ynorm = sum(y .* conj.(y))
        usec = usec = sqrt(xnorm * ynorm)
    else
        usec = 1
    end

    println(usec)
    return rawcorr / usec
end


function autocorr2(x, lag::Int)
    normx = sqrt(x' * conj.(x))
    # x = x / normx 
    return xcorr(x, x, lag)
end

frequency = 0.2
nLags = 1000
dt = 0.05
testfunction = 3*exp.(-1im*2*pi*frequency*(0:nLags)*dt)
testfunction2 = 7*exp.(-1im*2*pi*frequency*3.1*(0:nLags)*dt)

testfunction' * conj(testfunction)
statsversion = autocor(real.(testfunction), 0:100)



tests_plot = [autocorr2(testfunction, lag) for lag = 0:100]

plot(real(tests))
plot!(real(testfunction[1:101]) / 3)

# lag = 14
# rawcorr = y[(lag + 1):end]' * conj.(y[1:(end - lag)])

# rawcorr2 = sum(y[(lag + 1):end] .* conj.(y[1:(end - lag)]))

# plot(real(y[1:100]))

# plot(imag(testfunction[1:100]))


# plot!(real.(testfunction[1:101]) / 3)
# plot!(imag(tests))
# plot!(statsversion)


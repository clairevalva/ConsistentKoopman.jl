export 
    projectdata,
    reconstructfromproj,
    reconstruct

# TO DO: comment
function projectdata(x::AbstractVector, zetaR, mu, nEmb, nT_eig = nothing)
    # if want to match matlab, x needs to be cropped to correspond to the leave one out
    if nT_eig === nothing
        nT_eig = length(zetaR)
    end

    As = 1im*zeros(nEmb)
    for q = 1:nEmb
        As[q] = x[q:(nT_eig + q - 1)]' * conj.(zetaR) * mu
    end

    return As
end

function projectdata(x::AbstractMatrix, zetaR, mu, nEmb, nT_eig = nothing, nD = nothing)
    # if want to match matlab, x needs to be cropped to correspond to the leave one out
    if nT_eig === nothing
        nT_eig = length(zetaR)
    end

    if nD === nothing
        nD = size(x, 1)
    end

    As = 1im*zeros(nD, nEmb)
    for q = 1:nEmb
        As[:, q] = sum(x[:,q:(nT_eig + q - 1)] .* conj.(zetaR)'.* mu, dims = 2)
    end

    # not clear to me why it has to be like this to match but whatever, address later
    return conj.(As)
end

function reconstructfromproj(As::AbstractVector, zetaL, nEmb, nT_rec = nothing)
    # if want to match matlab, x needs to be cropped to correspond to the leave one out
    if nT_rec === nothing
        nT_rec = length(zetaL) + nEmb - 1
    end
    nT_eig = length(zetaL) 
    

    ytest = 1im*zeros(nT_rec)
    for j = 1:nT_rec
        if j < nEmb
            for k = 0:(j - 1)
                ytest[j] += As[1 + k] * zetaL[j - k]
            end
            ytest[j] /= j
        elseif j > nT_eig
            for k = (j - nT_eig):(nEmb - 1)
                ytest[j] += As[1 + k] * zetaL[j - k]
            end

            ytest[j] /= (nEmb - (j - nT_eig))
        else
            for k = 0:(nEmb - 1)
                ytest[j] += As[1 + k] * zetaL[j - k]
            end
            ytest[j] /= nEmb
        end
    end

    return ytest
end

function reconstructfromproj(As::AbstractMatrix, zetaL, nEmb, nT_rec = nothing, nD = nothing)
    # if want to match matlab, x needs to be cropped to correspond to the leave one out
    if nT_rec === nothing
        nT_rec = length(zetaL) + nEmb - 1
    end

    nT_eig = length(zetaL) 
    
    
    if nD === nothing
        nD = size(As, 1)
    end

    ytest = 1im*zeros(nD, nT_rec)
    for j = 1:nT_rec
        if j < nEmb
            for k = 0:(j - 1)
                ytest[:, j] .+= As[:,1 + k] * zetaL[j - k]
            end
            ytest[:, j] ./= j
        elseif j > nT_eig
            for k = (j - nT_eig):(nEmb - 1)
                ytest[:, j] .+= As[:,1 + k] * zetaL[j - k]
            end

            ytest[:, j] ./= (nEmb - (j - nT_eig))
        else
            for k = 0:(nEmb - 1)
                ytest[:, j] .+= As[:,1 + k] * zetaL[j - k]
            end
            ytest[:, j] ./= nEmb
        end
    end

    return ytest
end

function doreconstruction(x::AbstractVector, zetaR, zetaL, mu::Number, nEmb::Int)
    nT_eig = length(zetaR)
    nT_rec = nT_eig + nEmb - 1

    As = projectdata(x, zetaR, mu, nEmb, nT_eig)
    ytest = reconstructfromproj(As, zetaL, nEmb, nT_rec)

    return ytest
end

function doreconstruction(x::AbstractMatrix, zetaR, zetaL, mu::Number, nEmb::Int)
    nT_eig = length(zetaR)
    nT_rec = nT_eig + nEmb - 1
    nD = size(x, 1)

    As = projectdata(x, zetaR, mu, nEmb, nT_eig, nD)
    ytest = reconstructfromproj(As, zetaL, nEmb, nT_rec, nD)

    return ytest
end
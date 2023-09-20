export plotfuns

"""
    plotfuns(fs::Matrix, ts::Vector{Float64}, Nplot::Int64; pltimag::Bool = false, title::String = NaN, subtitles::Vector{String} = NaN  )

    function to symmetrize sparse kernel matrix as computed in sparseW()

        on June 14 I reversed the equality, to chekc later

    Arguments
    =================
    - fs: matrix of size nT × L where nT is the time axis and L is the number of functions
    - ts: vector of size nT which are time units
    - Nplot: no greater than L, how many to plot
    - pltimag: plot imaginary part of function?
    - title: sup title of plot
    - subtitles: subtitles of indiv plots, of length Nplot

"""

function plotfuns(fs::Matrix, ts::Vector{Float64}, Nplot::Int64;pltimag::Bool = false, title::String = "", subtitles::Vector{String} = [""])
    if length(subtitles) < Nplot
        subtitles = ["" for j in 1:Nplot]
    end

    splots = Vector()
    for n in 1:Nplot
        p = plot(ts, real(fs[:, n]), title = subtitles[n], legend = false)
        if pltimag
            plot!(ts, imag(fs[:, n]), legend = false)
        end
        push!(splots, p)
    end
    pbig = plot(splots..., plot_title = title,  layout = (Int(Nplot/2), 2))
    return pbig
end


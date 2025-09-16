image2tensor(image::AbstractMatrix{<:ImageCore.Colorant{<:Real,1}}) = image .|> ImageCore.RGB |> image2tensor
function image2tensor(image::AbstractMatrix{<:ImageCore.Colorant})
    @pipe image |>
    ImageCore.float32.(_) |>
    ImageCore.channelview(_) |>
    permutedims(_, (3,2,1))
end

function tensor2image(tensor::AbstractArray{T,3}; bands=[1,2,3]) where {T<:Real}
    @argcheck length(bands) == 3
    @pipe tensor |>
    _[:,:,bands] |>
    permutedims(_, (3,2,1)) |> 
    ImageCore.colorview(ImageCore.RGB{T}, _)
end

vec2array(x::AbstractVector, to::AbstractArray{T}, dim::Int) where T = vec2array(T.(x), to, dim)
function vec2array(x::AbstractVector{T}, to::AbstractArray{T,N}, dim::Int) where {T,N}
    @argcheck 0 < dim <= N
    @argcheck size(to, dim) == length(x)
    newshape = ntuple(i -> i == dim ? length(x) : 1, N)
    return reshape(x, newshape)
end

function unsqueeze(x::AbstractArray{<:Any,N}, dim::Int) where N
    newsize = (size(x)[1:dim-1]..., 1, size(x)[dim:end]...)
    return reshape(x, newsize)
end

function _quantiles(x::AbstractArray{<:Real}, lower, upper, dim::Int)
    map(1:size(x, dim)) do i
        data = selectdim(x, dim, i) |> vec
        return _quantiles(data, lower, upper)
    end
end

function _quantiles(x::AbstractVector{<:Real}, lower, upper)
    # Build Histogram
    edges, counts = @pipe build_histogram(x, 10000)

    # Get Percentiles
    ps = cumsum(counts) ./ sum(counts)

    # Get Lower and Upper Bounds
    lb = @pipe findfirst(>=(lower), ps) |> clamp(_, 1, 10000) |> edges[_]
    ub = @pipe findfirst(>=(upper), ps) |> clamp(_, 1, 10000) |> edges[_]
    return (lb, ub)
end

function build_histogram(img, nbins)
    minval = minimum(img)
    maxval = maximum(img)
    edges = _partition_interval(nbins, minval, maxval)
    return _build_histogram(img, edges)
end

function _build_histogram(img, edges::AbstractRange)
    lb = first(axes(edges,1))
    ub = last(axes(edges,1))
    first_edge, last_edge = first(edges), last(edges)
    inv_step_size = 1/step(edges)
    counts = fill(0, lb:ub)
    @inbounds for val in img
        if isnan(val) || ismissing(val) || isnothing(val)
            continue
        elseif val >= last_edge
            counts[ub] += 1
        else
            index = floor(Int, ((val-first_edge)*inv_step_size)) + 1
            counts[index] += 1
        end
    end
    edges, counts
end

function _partition_interval(nbins::Integer, minval::Real, maxval::Real)
    return range(minval, step=(maxval - minval) / nbins, length=nbins)
end

function _linear_stretch(x::AbstractArray{<:Real,N}) where N
    lb = minimum(x, dims=(1:N-1...,))
    ub = maximum(x, dims=(1:N-1...,))
    return clamp!((x .- lb) ./ (ub .- lb), 0, 1)
end

function scaled_map(f::Function, x::AbstractArray{T,N}; clamp=true) where {T,N}
    # Scale Values to the Range [0,1]
    lb = minimum(x, dims=(1:N-1...,))
    ub = maximum(x, dims=(1:N-1...,))
    scaled_x = (x .- lb) ./ (ub .- lb)

    # Apply f
    transformed_scaled_x = f(scaled_x)

    # Clamp Values
    if clamp
        clamp01!(transformed_scaled_x)
    end

    # Restore Original Scale
    return (transformed_scaled_x .* (ub .- lb)) .+ lb
end

rng_from_seed(seed::Int) = Random.MersenneTwister(seed)

apply_random(f, seed::Int, p::Float64, x) = roll_dice(seed, p) ? f(x) : x

function roll_dice(rng, p::Float64)
    @assert 0 <= p <= 1 "p must be between 0 and 1!"
    outcome = rand(rng, Random.uniform(Float64))
    return outcome <= p
end

function random_val(seed::Int, lower, upper)
    span = upper - lower
    outcome = rand(rng_from_seed(seed), Random.uniform(Float64))
    return round(Int, outcome * span) + lower
end

function channel_mean(x::AbstractArray{<:Real,N}, channeldim) where N
    dims = filter(x -> x !== channeldim, ntuple(identity, N))
    return mean(x; dims)
end

function channel_std(x::AbstractArray{<:Real,N}, channeldim) where N
    dims = filter(x -> x !== channeldim, ntuple(identity, N))
    return std(x; dims)
end

function pixel_extrema(x)
    lb = min(0, minimum(x))
    ub = max(1, maximum(x))
    return (lb, ub)
end

clamp01!(x) = clamp!(x, 0, 1)

function clamp_values!(new, old)
    lb, ub = pixel_extrema(old)
    return clamp!(new, lb, ub)
end

imsize(x::AbstractArray{T,N}) where {T,N} = size(x)[1:N-1]

function exclude_dim(ndims::Integer, dim::Integer)
    @assert 1 <= ndims
    @assert 1 <= dim <= ndims
    return ntuple(i -> i >= dim ? i+1 : i, ndims-1)
end

# Adapted from https://github.com/JuliaGraphics/Colors.jl/blob/master/src/conversions.jl
function rgb_to_hsv(x::AbstractVector{T}) where {T <: Real}
    # Find Minimum and Maximum Channels
    min_val, min_index = findmin(x)
    max_val, max_index = findmax(x)

    # Grayscale
    s0 = max_val - min_val
    s0 == zero(T) && return [zero(T), zero(T), max_val]

    # Compute Saturation
    s = s0 / max_val

    # Compute Hue
    diff = (max_index == 1 ? x[2]-x[3] : (max_index == 2 ? x[3]-x[1] : x[1]-x[2]))
    ofs = (max_index == 1 ? (x[2]<x[3])*T(360) : (max_index == 2 ? T(120) : T(240)))
    h0 = diff * T(60) / s0
    
    # Return HCV
    return [h0 + ofs, s, max_val]
end

function rgb_to_hsv(x::AbstractArray{T,3}) where {T <: Real}
    @assert size(x,3) == 3
    @assert all(x -> 0.0 <= x <= 1.0, x)
    return mapslices(rgb_to_hsv, x; dims=3)
end

# Adapted from https://github.com/JuliaGraphics/Colors.jl/blob/master/src/conversions.jl
function hsv_to_rgb(x::AbstractVector{T}) where {T <: Real}
    h = T(x[1] / 60)
    s = T(clamp(x[2], 0, 1))
    v = T(clamp(x[3], 0, 1))

    hi = unsafe_trunc(Int32, h) # instead of floor
    i = h < 0 ? hi - one(hi) : hi
    f = i & one(i) == zero(i) ? 1 - (h - i) : h - i
    im = 0x1 << (mod6(UInt8, i) & 0x07)
    
    # use `@fastmath` just to reduce the estimated costs for inlining
    @fastmath m = v * (1 - s)
    @fastmath n = v * (1 - s * f)

    rgb = _hsx_to_rgb(im, v, n, m)
    T <: ImageCore.FixedPoint && typemax(T) >= 1 ? rgb .% T : rgb
end

function hsv_to_rgb(x::AbstractArray{T,3}) where {T <: Real}
    @assert size(x,3) == 3
    mapslices(hsv_to_rgb, x; dims=3)
end

# Taken from https://github.com/JuliaGraphics/Colors.jl/blob/master/src/utilities.jl
function mod6(::Type{T}, x::Int32) where T
    return unsafe_trunc(T, x - 6 * ((widemul(x, 0x2aaaaaaa) + Int64(0x20000000)) >> 0x20))
end

# Adapted from https://github.com/JuliaGraphics/Colors.jl/blob/master/src/conversions.jl
function _hsx_to_rgb(im::UInt8, v, n, m)
    #=
    if     hue <  60; im = 0b000001 # ---------+
    elseif hue < 120; im = 0b000010 # --------+|
    elseif hue < 180; im = 0b000100 # -------+||
    elseif hue < 240; im = 0b001000 # ------+|||
    elseif hue < 300; im = 0b010000 # -----+||||
    else            ; im = 0b100000 # ----+|||||
    end                             #     ||||||
    (hue < 60 || hue >= 300) === ((im & 0b100001) != 0x0)
    =#
    r = ifelse((im & 0b100001) == 0x0, ifelse((im & 0b010010) == 0x0, m, n), v)
    g = ifelse((im & 0b000110) == 0x0, ifelse((im & 0b001001) == 0x0, m, n), v)
    b = ifelse((im & 0b011000) == 0x0, ifelse((im & 0b100100) == 0x0, m, n), v)
    return [r, g, b]
end
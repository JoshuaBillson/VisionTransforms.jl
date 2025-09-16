"""
    linear_stretch(x::AbstractArray, lower::Vector{<:Real}, upper::Vector{<:Real}, channel_dim::Int)

Perform a linear histogram stretch on `x` such that `lower` is mapped to 0 and `upper` is mapped to 1.
Values outside the interval `[lower, upper]` will be clamped.
"""
function linear_stretch(x::AbstractArray{<:Real,N}, lower::Vector{<:Real}, upper::Vector{<:Real}, channel_dim::Int) where N
    @argcheck 1 <= channel_dim <= N
    lower = vec2array(lower, x, channel_dim)
    upper = vec2array(upper, x, channel_dim)
    return clamp!((x .- lower) ./ (upper .- lower), 0, 1)
end

"""
    per_image_linear_stretch(x::AbstractArray, lower::Real, upper::Real, channel_dim::Int)

Apply a linear stretch to scale all values to the range [0, 1]. The arguments `lower` and
`upper` specify the percentiles at which to define the lower and upper bounds from each channel
in the source image. Values that either fall below `lower` or above `upper` will be clamped.
"""
function per_image_linear_stretch(x::AbstractArray{<:Real,N}, lower::Real, upper::Real, channel_dim::Int) where N
    @argcheck 0 <= lower <= upper <= 1
    @argcheck 1 <= channel_dim <= N
    dims = exclude_dim(N, channel_dim)
    mapslices(x, dims=dims) do x
        data = vec(x) |> collect |> sort!
        lb = quantile(data, 0.02, sorted=true)
        ub = quantile(data, 0.98, sorted=true)
        return clamp!((x .- lb) ./ (ub .- lb), 0, 1)
    end
end

"""
    normalize(x::AbstractArray{<:Number,N}, μ::AbstractVector, σ::AbstractVector; channeldim=N)

Normalize the input array with respect to the specified dimension so that the mean is 0
and the standard deviation is 1.

# Parameters
- `x`: A tensor containing one or more 2D or 3D images or 2D image series.
- `μ`: A `Vector` of means for each index in `channeldim`.
- `σ`: A `Vector` of standard deviations for each index in `channeldim`.
- `channeldim`: The dimension corresponding to channels in `x`.
"""
normalize(x::AbstractArray{<:Integer}, args...; kw...) = normalize(Float32.(x), args...; kw...)
function normalize(x::AbstractArray{T,N}, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}; channeldim=N) where {T<:AbstractFloat,N}
    @argcheck 1 <= channeldim <= N
    @argcheck length(μ) == length(σ) == size(x,channeldim)
    return (x .- vec2array(μ, x, channeldim)) ./ vec2array(σ, x, channeldim)
end

"""
    per_image_normalize(x::AbstractArray{<:Number,N}; channeldim=N, obsdim=nothing)

Normalize the input array so that the mean and standard deviation of each channel is 0 and 1, respectively.
Unlike `normalize`, this method will compute new statistics for each image in `x`. This is more computationally
expensive, but may be more suitable when there is significant domain shift between train and test images. 

# Parameters
- `x`: A tensor containing one or more 2D or 3D images or 2D image series.
- `channeldim`: The dimension corresponding to the image channel.
- `obsdim`: The dimension corresponding to observations if present.
"""
per_image_normalize(x::AbstractArray{<:Integer}; kw...) = per_image_normalize(Float32.(x); kw...)
function per_image_normalize(x::AbstractArray{<:AbstractFloat,N}; channeldim::Integer=N, obsdim=nothing) where N
    # Validate Arguments
    @argcheck 1 <= channeldim <= N
    @argcheck isnothing(obsdim) || (isinteger(obsdim) && (1 <= obsdim <= N))

    # Determine Dims Over Which to Compute Statistics
    dims = isnothing(obsdim) ? exclude_dim(N, channeldim) : filter(!=(obsdim), exclude_dim(N, channeldim))

    # Normalize Each Channel
    return (x .- mean(x; dims)) ./ std(x; dims)
end

"""
    blur(x::AbstractArray{<:Real}, strength::Tuple; ndims::Integer=2)
    blur(x::AbstractArray{<:Real}, strength::Number; ndims::Integer=2)

Blur the image `x` by applying a gaussian filter with a standard deviation of `strength`.
"""
blur(x::AbstractArray{<:Real}, strength::Number; ndims::Integer=2) = blur(x, ntuple(_ -> strength, ndims); ndims)
function blur(x::AbstractArray{<:Real}, strength::Tuple; ndims::Integer=2)
    kernel = ImageFiltering.Kernel.gaussian(strength)
    return mapslices(x -> ImageFiltering.imfilter(x, kernel), x; dims=ntuple(identity, ndims))
end

"""
    sharpen(x::AbstractArray, strength::Real, channeldim::Int)

Sharpen the image `x` by applying a high-frequency-boosting filter.
"""
function sharpen(x::AbstractArray{<:Real,N}, strength::Real) where N
    kernel = ImageFiltering.centered([-1 -1 -1; -1 8 -1; -1 -1 -1])
    edges = mapslices(x -> ImageFiltering.imfilter(x, kernel), x, dims=(1,2))
    return clamp_values!(strength .* edges .+ x, x)
end

"""
    posterize(x::AbstractArray, nbits::Int)

Posterize an image `x` by reducing the number of bits for each color channel to `nbits`.
"""
function posterize(x::AbstractArray{T}, nbits::Int) where {T <: Real}
    @argcheck 1 <= nbits <= 8
    lb, ub = pixel_extrema(x)
    stretched = (x .- lb) ./ (ub - lb)
    posterized = ImageCore.Normed{UInt8,nbits}.(stretched)
    return T.((posterized .* (ub - lb)) .+ lb)
end

"""
    add_noise(seed::Integer, dist::Distributions.Distribution, x::AbstractArray{<:Real,N}; channeldim=N, correlated=true)

Add noise generated by the distribution `dist` to the image tensor `x`.

# Parameters
- `seed`: A seed to make the outcome reproducible.
- `dist`: A `Distributions.Distribution` object from which to sample the noise.
- `x`: A tensor containing a 2D or 3D image or image series.
- `correlated`: If true, applies the same noise value to each channel in the image.
"""
function add_noise(seed::Integer, dist::Distributions.Distribution, x::AbstractArray{T,N}; channeldim=N, correlated=true) where {T <: Real, N}
    noise_dim = ntuple(i -> ((i == channeldim) && correlated) ? 1 : size(x,i), N)
    noise = rand(rng_from_seed(seed), dist, noise_dim) .|> T
    return x .+ noise
end

"""
    multiply_noise(seed::Integer, dist::Distributions.Distribution, x::AbstractArray{<:Real,N}; channeldim=N, correlated=true)

Multiply noise generated by the distribution `dist` to the image tensor `x`.

# Parameters
- `seed`: A seed to make the outcome reproducible.
- `dist`: A `Distributions.Distribution` object from which to sample the noise.
- `x`: A tensor containing a 2D or 3D image or image series.
- `correlated`: If true, applies the same noise value to each channel in the image.
"""
function multiply_noise(seed::Integer, dist::Distributions.Distribution, x::AbstractArray{T,N}; channeldim=N, correlated=true) where {T <: Real, N}
    noise_dim = ntuple(i -> ((i == channeldim) && correlated) ? 1 : size(x,i), N)
    noise = rand(rng_from_seed(seed), dist, noise_dim) .|> T
    return x .* noise
end
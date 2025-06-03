image2tensor(image::AbstractMatrix{<:ImageCore.Colorant{<:Real,1}}) = image .|> ImageCore.RGB |> image2tensor
function image2tensor(image::AbstractMatrix{<:ImageCore.Colorant})
    @pipe image |>
    ImageCore.float32.(_) |>
    ImageCore.channelview(_) |>
    permutedims(_, (3,2,1))
end

function tensor2image(tensor::AbstractArray{<:Real,3}; bands=[1,2,3])
    @argcheck length(bands) == 3
    @pipe tensor |>
    selectdim(_, 3, bands) |> 
    ImageCore.n0f8.(_) |> 
    permutedims(_, (3,2,1)) |> 
    ImageCore.colorview(ImageCore.RGB, _)
end

"""
    imresize(img::AbstractArray, sz::Tuple, method::Symbol)

Resize `img` to `sz` with the specified resampling `method`.

# Parameters
- `img`: The image to be resized.
- `sz`: The width and height of the output as a tuple.
- `method`: Either `:nearest` or `:bilinear`.
"""
function imresize(img::AbstractArray{<:Real,N1}, sz::NTuple{N2,Int}, method::Symbol) where {N1,N2}
    @argcheck N1 >= N2
    @argcheck method in (:nearest, :bilinear)
    @argcheck all(dim -> size(img, dim) > 1, eachindex(sz))
    return mapslices(x -> _imresize(x, sz, method), img; dims=ntuple(identity, N2))
end

function _imresize(img::AbstractArray, sz::Tuple, method::Symbol)
    @match method begin
        :nearest => ImageTransformations.imresize(img, sz, method=Constant())
        :bilinear => ImageTransformations.imresize(img, sz, method=Linear())
    end
end

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
    crop(x, sz::Int, ul=(1,1))
    crop(x, sz::Tuple{Int,Int}, ul=(1,1))

Crop a tile equal to `size` out of `x` with an upper-left corner defined by `ul`.
"""
crop(x::AbstractArray, sz::Int, ul=(1,1)) = crop(x, (sz, sz), ul)
function crop(x::AbstractArray, sz::Tuple{Int,Int}, ul=(1,1))
    # Arg Checks
    @argcheck length(sz) == length(ul)
    @argcheck all(1 .<= sz .<= size(x)[1:2])
    @argcheck all(1 .<= ul .<= size(x)[1:2])

    # Compute Lower-Right Coordinates
    lr = ul .+ sz .- 1

    # Check Bounds
    any(lr .> imsize(x)) && throw(ArgumentError("Crop is out of bounds!"))

    # Crop Tile
    return crop_image(x, ul[1]:lr[1], ul[2]:lr[2])
end

"""
    center_crop(x::AbstractArray, sz::Int)
    center_crop(x::AbstractArray, sz::Tuple{Int,Int})

Crop `x` to the size specified by `sz` from the center.
"""
center_crop(x::AbstractArray, sz::Int) = center_crop(x, (sz,sz))
function center_crop(x::AbstractArray, sz::Tuple{Int,Int})
    @argcheck all(1 .<= sz .<= imsize(x))
    pad = imsize(x) .- sz
    ul = (pad .÷ 2) .+ 1
    return crop(x, sz, ul)
end

"""
    random_crop(seed::Int, x::AbstractArray, sz::Tuple{Int,Int})

Crop a randomly placed tile equal to `sz` from the array `x`.
"""
function random_crop(seed::Int, x::AbstractArray, sz::Tuple{Int,Int})
    @argcheck all(sz .>= 1)
    lower_bounds = first.(axes(x))[1:2]
    upper_bounds = last.(axes(x))[1:2] .- sz .+ 1
    ul = random_point(seed, lower_bounds, upper_bounds)
    return crop(x, sz, ul)
end

"""
    center_zoom(x::AbstractArray, zoom_strength::Int, method::Symbol)

Zoom to the center of `x` by a factor of `zoom_strength`.
"""
function center_zoom(x::AbstractArray, zoom_strength::Int, method::Symbol)
    @argcheck zoom_strength >= 1
    newsize = imsize(x) .÷ zoom_strength
    return imresize(center_crop(x, newsize), imsize(x), method)
end

"""
    random_zoom(seed::Integer, x::AbstractArray, zoom_strength::Real, method::Symbol)

Zoom to a random location in `x` by a factor of `zoom_strength`.
"""
function random_zoom(seed::Integer, x::AbstractArray, zoom_strength::Real, method::Symbol)
    @argcheck zoom_strength >= 1
    newsize = round.(Int, imsize(x) .÷ zoom_strength)
    return imresize(random_crop(seed, x, newsize), imsize(x), method)
end

"""
    flipX(x)

Flip the image `x` across the horizontal axis.
"""
flipX(x::AbstractArray) = selectdim(x, 2, size(x,2):-1:1)

"""
    flipY(x)

Flip the image `x` across the vertical axis.
"""
flipY(x::AbstractArray) = selectdim(x, 1, size(x,1):-1:1)

"""
    rot90(x)

Rotate the image `x` by 90 degress. 
"""
rot90(x::AbstractArray{<:Any,N}) where N = permutedims(x, (2,1,3:N...)) |> flipX


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
    grayscale(x::AbstractArray, channeldim::Int)

Convert `x` to a grayscale image.
"""
function grayscale(x::AbstractArray{<:Real,N}, channeldim::Int) where N
    repeatdims = ntuple(i -> i == channeldim ? size(x,channeldim) : 1, N)
    return repeat(mean(x, dims=channeldim), repeatdims...)
end

"""
    adjust_contrast(x::AbstractArray, contrast::Real, channeldim::Int)

Adjust the contrast of `x` by `contrast`.
"""
function adjust_contrast(x::AbstractArray{T}, contrast::Real) where {T<:Real}
    @argcheck 0 < contrast
    return clamp_values!(x .* T(contrast), x)
end

"""
    adjust_brightness(x::AbstractArray, brightness::Real)

Adjust the brightness of `x` by `brightness`.
"""
function adjust_brightness(x::AbstractArray{T}, brightness::Real) where {T <: Real}
    return clamp_values!(x .+ T(brightness), x)
end

"""
    shift_hue(x::AbstractArray, shift::Real, channeldim::Int)

Shift the hue of `x` in the HSV color space by the amount specified by `shift`.
"""
function shift_hue(x::AbstractArray{T,N}, shift::Real, channeldim::Int) where {T<:Real,N}
    @argcheck size(x, channeldim) == 3
    @argcheck all(x -> 0 <= x <= 1, x)
    _shift = T.(reshape([shift, 0, 0], (1,1,3)))
    return @pipe rgb_to_hsv(x) |> ((_ .+ _shift) .% 360) |> hsv_to_rgb
end

"""
    color_jitter(seed::Int, x::AbstractArray, strength::Int, channeldim::Int)

Applies random color jittering transformations (hue, saturation, and brightness) to 
the input image `x`.

# Parameters
- `seed`: A seed to make the outcome reproducible.
- `x`: A tensor containing an RGB image or image series.
- `strength`: The strength of the jittering, from 1 to 10.
- `channeldim`: The dimension corresponding to channels in `x`.
"""
function color_jitter(seed::Int, x::AbstractArray{T}, strength::Int, channeldim::Int) where {T <: Real}
    @argcheck 1 <= strength <= 10
    @argcheck size(x, channeldim) == 3
    rng = MersenneTwister(seed)
    hue_shift = LinRange(20,180,10)[strength] * rand(rng, [-1,1])
    saturation_shift = LinRange(0.05,0.20,10)[strength] * rand(rng, [-1,1])
    value_shift = LinRange(0.05,0.20,10)[strength] * rand(rng, [-1,1])
    shift = vec2array([hue_shift, saturation_shift, value_shift], x, channeldim)
    @pipe rgb_to_hsv(x) |> ((_ .+ shift) .% 360) |> hsv_to_rgb
end

"""
    permute_channels(x::AbstractArray, channeldim::Int)

Permute the channel ordering of `x`.
"""
function permute_channels(x::AbstractArray, channeldim::Int)
    return selectdim(x, channeldim, Random.randperm(size(x,channeldim)))
end

"""
    invert_color(x::AbstractArray{<:Real})

Invert the colors of `x`.
"""
function invert(x::AbstractArray{T}) where {T <: Real}
    lb, ub = pixel_extrema(x)
    midpoint = (lb + ub) / 2
    return T(2 * midpoint) .- x  # (midpoint - x) + midpoint = 2 * midpoint - x
end

"""
    solarize(x::AbstractArray{<:Real}; threshold=0.75)

Solarize `x` by inverting all values above `threshold`.
"""
function solarize(x::AbstractArray{<:Real}; threshold=0.75)
    lb, ub = pixel_extrema(x)
    thresh = ((ub - lb) * threshold) + lb
    return ifelse.(x .> thresh, invert(x), x)
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
    noise = rand(Random.MersenneTwister(seed), dist, noise_dim) .|> T
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
    noise = rand(Random.MersenneTwister(seed), dist, noise_dim) .|> T
    return x .* noise
end
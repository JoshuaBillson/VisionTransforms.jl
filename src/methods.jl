image2tensor(image::AbstractMatrix{<:Images.Colorant{<:Real,1}}) = image .|> Images.RGB |> image2tensor
function image2tensor(image::AbstractMatrix{<:Images.Colorant})
    @pipe image |>
    Images.float32.(_) |>
    Images.channelview(_) |>
    permutedims(_, (3,2,1)) |>
    _putobs(_)
end

_putobs(x::AbstractArray) = reshape(x, size(x)..., 1)

function tensor2image(tensor::AbstractArray{<:Real,4}; bands=[1,2,3])
    @argcheck length(bands) == 3
    if size(tensor, 4) > 1
        return map(i -> tensor2image(selectdim(tensor, 4, i:i); bands=bands), axes(tensor)[end])
    else
        return @pipe tensor |>
        selectdim(_, 4, 1) |> 
        selectdim(_, 3, bands) |> 
        Images.n0f8.(_) |> 
        permutedims(_, (3,2,1)) |> 
        Images.colorview(Images.RGB, _)
    end
end

function raster2tensor(x::Rasters.AbstractRaster)
    raster_dims = (Rasters.X,Rasters.Y,Rasters.Z,Rasters.Band,Rasters.Ti)
    @argcheck Rasters.hasdim(x, Rasters.X)
    @argcheck Rasters.hasdim(x, Rasters.Y)
    @argcheck all(dim -> dim in Rasters.name(raster_dims), Rasters.name(Rasters.dims(x)))

    # Handle Missing Band Dimension
    if !Rasters.hasdim(x, Rasters.Band)  # Add Missing Band Dim
        return raster2tensor(_putdim(x, Rasters.Band))
    end

    # Enforce (X,Y,Z,Band,Ti) Order
    _dims = Rasters.commondims(raster_dims, Rasters.dims(x))  
    x = _permute(x, _dims)

    # Return Raster Data as Tensor
    return parent(x) |> _putobs
end

function _putdim(raster::Rasters.AbstractRaster, ::Type{T}) where {T <: Rasters.DD.Dimension}
    newdims = (Rasters.dims(raster)..., T(Base.OneTo(1))::T{Base.OneTo{Int64}})
    return Rasters.Raster(reshape(raster.data, (size(raster)..., 1)), newdims)
end

function _permute(x::Rasters.AbstractRaster, dims)
    if Rasters.name(Rasters.dims(x)) == Rasters.name(dims)
        return x
    end
    return permutedims(x, dims)
end

"""
    imresize(img::AbstractArray, sz::Tuple{Int,Int}; method=:bilinear)

Resize `img` to `sz` with the specified resampling `method`.

# Parameters
- `img`: The image to be resized.
- `sz`: The width and height of the output as a tuple.
- `method`: Either `:nearest` or `:bilinear`.
"""
imresize(img::Mask2D, sz::Tuple) = imresize(img.data, sz, :nearest, 3) |> Mask2D
imresize(img::Mask3D, sz::Tuple) = imresize(img.data, sz, :nearest, 4) |> Mask3D
imresize(img::Image2D, sz::Tuple) = imresize(img.data, sz, :bilinear, 3) |> Image2D
imresize(img::Image3D, sz::Tuple) = imresize(img.data, sz, :bilinear, 4) |> Image3D
imresize(img::Series2D, sz::Tuple) = imresize(img.data, sz, :bilinear, 3) |> Series2D
function imresize(img::AbstractArray{<:Real,N}, sz::Tuple, method::Symbol, channeldim::Int) where {N}
    @argcheck method in (:nearest, :bilinear)
    @argcheck channeldim < N

    # Iterate Over Observations
    dst = similar(img, _newsize(sz,img))
    for obs in 1:size(img,N)
        _dst = selectdim(dst, N, obs)
        _img = selectdim(img, N, obs)

        # Handle Singleton Channels
        if size(img, channeldim) == 1
            _dst = selectdim(_dst, channeldim, 1)
            _img = selectdim(_img, channeldim, 1)
        end

        # Resize Image
        _dst .= _imresize(_img, sz, method)
    end

    return dst
end

_newsize(sz::Tuple{Int,Int}, x::AbstractArray{<:Any,N}) where N = (sz[1], sz[2], size(x)[3:N]...)
_newsize(sz::Tuple{Int,Int,Int}, x::AbstractArray{<:Any,N}) where N = (sz[1], sz[2], sz[3], size(x)[4:N]...)

function _imresize(img::AbstractArray, sz::Tuple, method::Symbol)
    @match method begin
        :nearest => Images.imresize(img, sz, method=Constant())
        :bilinear => Images.imresize(img, sz, method=Linear())
    end
end

"""
    linear_stretch(x::Image2D, lower::Vector{<:Real}, upper::Vector{<:Real})
    linear_stretch(x::Image3D, lower::Vector{<:Real}, upper::Vector{<:Real})
    linear_stretch(x::Series2D, lower::Vector{<:Real}, upper::Vector{<:Real})
    linear_stretch(x::AbstractArray, lower::Vector{<:Real}, upper::Vector{<:Real}, channel_dim::Int)

Perform a linear histogram stretch on `x` such that `lower` is mapped to 0 and `upper` is mapped to 1.
Values outside the interval `[lower, upper]` will be clamped.
"""
linear_stretch(x::Image2D, lower::Vector{<:Real}, upper::Vector{<:Real}) = linear_stretch(x, lower, upper, 3)
linear_stretch(x::Image3D, lower::Vector{<:Real}, upper::Vector{<:Real}) = linear_stretch(x, lower, upper, 4)
linear_stretch(x::Series2D, lower::Vector{<:Real}, upper::Vector{<:Real}) = linear_stretch(x, lower, upper, 3)
function linear_stretch(x::AbstractArray{<:Real,N}, lower::Vector{<:Real}, upper::Vector{<:Real}, channel_dim::Int) where N
    @argcheck 1 <= channel_dim <= N
    lower = vec2array(lower, x, channel_dim)
    upper = vec2array(upper, x, channel_dim)
    return clamp!((x .- lower) ./ (upper .- lower), 0, 1)
end

"""
    per_image_linear_stretch(x::Image2D, lower::Real, upper::Real)
    per_image_linear_stretch(x::Image3D, lower::Real, upper::Real)
    per_image_linear_stretch(x::Series2D, lower::Real, upper::Real)
    per_image_linear_stretch(x::AbstractArray, lower::Real, upper::Real, channel_dim::Int)

Apply a linear stretch to scale all values to the range [0, 1]. The arguments `lower` and
`upper` specify the percentiles at which to define the lower and upper bounds from each channel
in the source image. Values that either fall below `lower` or above `upper` will be clamped.
"""
per_image_linear_stretch(x::Image2D, lower::Real, upper::Real) = per_image_linear_stretch(x, lower, upper, 3)
per_image_linear_stretch(x::Image3D, lower::Real, upper::Real) = per_image_linear_stretch(x, lower, upper, 4)
per_image_linear_stretch(x::Series2D, lower::Real, upper::Real) = per_image_linear_stretch(x, lower, upper, 3)
function per_image_linear_stretch(x::AbstractArray{<:Real,N}, lower::Real, upper::Real, channel_dim::Int) where N
    @argcheck 0 <= lower <= upper <= 1
    @argcheck 1 <= channel_dim <= N
    dims = filter(x -> x != channel_dim && x != N, ntuple(identity, N))
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
    @argcheck all(sz .>= 1)
    @argcheck all(1 .<= ul .<= _size(x))

    # Compute Lower-Right Coordinates
    lr = ul .+ sz .- 1

    # Check Bounds
    any(lr .> _size(x)) && throw(ArgumentError("Crop is out of bounds!"))

    # Crop Tile
    return _crop(x, ul[1]:lr[1], ul[2]:lr[2])
end

_size(x::AbstractArray)::Tuple{Int,Int} = size(x)[1:2]

_crop(x::AbstractArray{<:Any,2}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims]
_crop(x::AbstractArray{<:Any,3}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:]
_crop(x::AbstractArray{<:Any,4}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:]
_crop(x::AbstractArray{<:Any,5}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:,:]
_crop(x::AbstractArray{<:Any,6}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:,:,:]

"""
    center_crop(x::AbstractArray, sz::Int)
    center_crop(x::AbstractArray, sz::Tuple{Int,Int})

Crop `x` to the size specified by `sz` from the center.
"""
center_crop(x::AbstractArray, sz::Int) = center_crop(x, (sz,sz))
function center_crop(x::AbstractArray, sz::Tuple{Int,Int})
    @argcheck all(1 .<= sz .<= _size(x))
    pad = _size(x) .- sz
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
    flipX(x)

Flip the image `x` across the horizontal axis.
"""
flipX(x::AbstractArray{<:Any,2}) = x[:,end:-1:1]
flipX(x::AbstractArray{<:Any,3}) = x[:,end:-1:1,:]
flipX(x::AbstractArray{<:Any,4}) = x[:,end:-1:1,:,:]
flipX(x::AbstractArray{<:Any,5}) = x[:,end:-1:1,:,:,:]
flipX(x::AbstractArray{<:Any,6}) = x[:,end:-1:1,:,:,:,:]

"""
    flipY(x)

Flip the image `x` across the vertical axis.
"""
flipY(x::AbstractArray{<:Any,2}) = x[end:-1:1,:]
flipY(x::AbstractArray{<:Any,3}) = x[end:-1:1,:,:]
flipY(x::AbstractArray{<:Any,4}) = x[end:-1:1,:,:,:]
flipY(x::AbstractArray{<:Any,5}) = x[end:-1:1,:,:,:,:]
flipY(x::AbstractArray{<:Any,6}) = x[end:-1:1,:,:,:,:,:]

"""
    rot90(x)

Rotate the image `x` by 90 degress. 
"""
rot90(x::AbstractArray{<:Any,4}) = @pipe permutedims(x, (2,1,3,4)) |> reverse(_, dims=2)
rot90(x::AbstractArray{<:Any,5}) = @pipe permutedims(x, (2,1,3,4,5)) |> reverse(_, dims=2)
rot90(x::AbstractArray{<:Any,6}) = @pipe permutedims(x, (2,1,3,4,5,6)) |> reverse(_, dims=2)

"""
    normalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=1)

Normalize the input array with respect to the specified dimension so that the mean is 0
and the standard deviation is 1.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to normalize the input array.
"""
normalize(x::AbstractArray{<:Integer}, args...) = normalize(Float32.(x), args...)
normalize(x::Image2D, μ::AbstractVector, σ::AbstractVector) = normalize(x, μ, σ, 3)
normalize(x::Image3D, μ::AbstractVector, σ::AbstractVector) = normalize(x, μ, σ, 4)
normalize(x::Series2D, μ::AbstractVector, σ::AbstractVector) = normalize(x, μ, σ, 3)
function normalize(x::AbstractArray{T,N}, μ::AbstractVector, σ::AbstractVector, dim::Int) where {T<:AbstractFloat,N}
    @argcheck 1 <= dim <= N
    @argcheck length(μ) == length(σ) == size(x,dim)
    return (x .- vec2array(T.(μ), x, dim)) ./ vec2array(T.(σ), x, dim)
end

"""
    per_image_normalize(x::Image2D)
    per_image_normalize(x::Image3D)
    per_image_normalize(x::Series2D)
    per_image_normalize(x::AbstractArray, dims::Tuple)

Normalize the input array so that the mean and standard deviation of each channel is 0 and 1, respectively.
Unlike `normalize`, this method will compute new statistics for each image in `x`. This is more computationally
expensive, but may be more suitable when there is significant domain shift between train and test images. 

# Parameters
- `x`: A tensor containing one or more 2D or 3D images or 2D image series.
- `dims`: The dimensions over which to compute image statistics.
"""
per_image_normalize(x::Image2D) = per_image_normalize(x, (1,2))
per_image_normalize(x::Image3D) = per_image_normalize(x, (1,2,3))
per_image_normalize(x::Series2D) = per_image_normalize(x, (1,2,4))
per_image_normalize(x::AbstractArray{<:Integer}, args...) = per_image_normalize(Float32.(x), args...)
function per_image_normalize(x::AbstractArray{<:AbstractFloat,N}, dims) where N
    μ = mean(x, dims=dims)
    σ = std(x, dims=dims)
    return (x .- μ) ./ σ
end

"""
    color_jitter([rng], x::Image2D, contrast, brightness; kw...)
    color_jitter([rng], x::Image3D, contrast, brightness; kw...)
    color_jitter([rng], x::Series2D, contrast, brightness; kw...)
    color_jitter([rng], x::AbstractArray, contrast, brightness, dims; usemax=true)

Applies random color jittering transformations (contrast and brightness adjustments) to 
input images or data series according to the formula `α * x + β * M`, where `α` is contrast,
`β` is brightness, and `M` is either the mean or maximum value of `x`.

# Parameters
- `rng`: A random number generator to make the outcome reproducible.
- `dist`: A `Distributions.Distribution` object from which to sample the noise.
- `x`: A tensor containing a 2D or 3D image or image series.
- `correlated`: If true, applies the same noise value to each channel in the image.
"""
color_jitter(x, contrast, brightness) = color_jitter(Random.default_rng(), x, contrast, brightness)
color_jitter(x, contrast, brightness, channeldim) = color_jitter(Random.default_rng(), x, contrast, brightness, channeldim)
color_jitter(rng::Random.AbstractRNG, x::Image2D, contrast, brightness) = color_jitter(rng, x, contrast, brightness, 3)
color_jitter(rng::Random.AbstractRNG, x::Image3D, contrast, brightness) = color_jitter(rng, x, contrast, brightness, 4)
color_jitter(rng::Random.AbstractRNG, x::Series2D, contrast, brightness) = color_jitter(rng, x, contrast, brightness, 3)
color_jitter(rng::Random.AbstractRNG, x::AbstractArray, contrast, brightness, channeldim) = _color_jitter(rng, x, contrast, brightness, channeldim)

function _color_jitter(rng::Random.AbstractRNG, x::AbstractArray{T,N}, contrast::AbstractVector{<:Real}, brightness::AbstractVector{<:Real}, channeldim::Int) where {T<:Real,N}
    # Compute Statistics
    dims = _fold_dims(x, channeldim)
    σ = std(x; dims)
    μ = mean(x; dims)

    # Apply Brightness and Contrast Adjustment
    rand_dim = ntuple(i -> i == N ? size(x,N) : 1, N)
    α = T.(rand(rng, contrast, rand_dim))
    β = T.(rand(rng, brightness, rand_dim) .* σ)
    x = ((x .- μ) .* α .+ μ .+ β)

    # Apply Color Jitter
    jitter_dim = ntuple(i -> i == channeldim ? size(x,channeldim) : 1, N)
    jitter = rand(-1.0:0.1:1.0, jitter_dim) .* σ
    return x .+ jitter
end

function _fold_dims(::AbstractArray{<:Any,N}, channeldim::Int)::NTuple{N-2,Int} where N
    return filter(x -> !(x in (channeldim, N)), ntuple(identity, N))
end

"""
    invert(x::AbstractArray{<:Real})

Invert the values of `x` according to the formula `maximum(x) .- x`.
"""
invert(x::AbstractArray{<:Real}) = maximum(x) .- x

"""
    solarize(x::AbstractArray{<:Real}; threshold=0.75)

Solarize `x` by inverting all values above `threshold`.
"""
solarize(x::AbstractArray{<:Real}; threshold=0.75) = ifelse.(x .> threshold, invert(x), x)

"""
    add_noise([rng], dist::Distributions.Distribution, x::AbstractArray; correlated=true)

Add noise generated by the distribution `dist` to the image tensor `x`.

# Parameters
- `rng`: A random number generator to make the outcome reproducible.
- `dist`: A `Distributions.Distribution` object from which to sample the noise.
- `x`: A tensor containing a 2D or 3D image or image series.
- `correlated`: If true, applies the same noise value to each channel in the image.
"""
add_noise(args...; kw...) = add_noise(Random.default_rng(), args...; kw...)
add_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::Image2D; kw...) = add_noise(rng, dist, x, 3; kw...)
add_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::Image3D; kw...) = add_noise(rng, dist, x, 4; kw...)
add_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::Series2D; kw...) = add_noise(rng, dist, x, 3; kw...)
function add_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::AbstractArray{T,N}, channeldim::Int; correlated=true) where {T <: Real, N}
    noise_dim = ntuple(i -> i == channeldim && correlated ? 1 : size(x,i), N)
    noise = rand(rng, dist, noise_dim) .|> T
    return x .+ noise
end

"""
    multiply_noise([rng], dist::Distributions.Distribution, x::AbstractArray; correlated=true)

Multiply noise generated by the distribution `dist` to the image tensor `x`.

# Parameters
- `rng`: A random number generator to make the outcome reproducible.
- `dist`: A `Distributions.Distribution` object from which to sample the noise.
- `x`: A tensor containing a 2D or 3D image or image series.
- `correlated`: If true, applies the same noise value to each channel in the image.
"""
multiply_noise(args...; kw...) = multiply_noise(Random.default_rng(), args...; kw...)
multiply_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::Image2D; kw...) = multiply_noise(rng, dist, x, 3; kw...)
multiply_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::Image3D; kw...) = multiply_noise(rng, dist, x, 4; kw...)
multiply_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::Series2D; kw...) = multiply_noise(rng, dist, x, 3; kw...)
function multiply_noise(rng::Random.AbstractRNG, dist::Distributions.Distribution, x::AbstractArray{T,N}, channeldim::Int; correlated=true) where {T <: Real, N}
    noise_dim = ntuple(i -> i == channeldim && correlated ? 1 : size(x,i), N)
    noise = rand(rng, dist, noise_dim) .|> T
    return x .* noise
end

_channeldim(::Type{<:Image2D}) = 3
_channeldim(::Type{<:Image3D}) = 4
_channeldim(::Type{<:Series2D}) = 3
import Base.|>

abstract type AbstractTransform end

apply(::AbstractTransform, x::NoOp, ::Int) = x
apply(::AbstractTransform, x::DType, ::Int) = x
function apply(::AbstractTransform, x::T, ::Int) where T
    throw(ArgumentError("Transform expects input to sub-type `DType`, but received $T."))
end

"""
    transform(t::AbstractTransform, dtype, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
transform(::AbstractTransform, ::Type{<:NoOp}, x)  = x
transform(t::AbstractTransform, ::Type{T}, x) where {T <: DType} = apply(t, T(x), rand(1:1000)) |> parent
function transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)
    seed = rand(1:1000)
    map(zip(dtypes, x)) do (T, x)
        apply(t, T(x), seed) |> parent
    end
end

"""
    Resize(sz::Tuple)

Resample `x` according to the specified `scale`. `Mask` types will always be
resampled with `:near` interpolation, whereas `Images` will be resampled with 
either `:bilinear` (`scale` > `1`) or `:average` (`scale` < `1`).

# Parameters
- `x`: The image/mask to be resampled.
- `sz`: The size of the output image.
"""
struct Resize{S<:Tuple} <: AbstractTransform
    sz::S
end

apply(t::Resize, x::Union{<:AbstractMask,<:AbstractImage}, ::Int) = imresize(x, t.sz)

description(x::Resize) = "Resize to $(x.sz)."

"""
    Scale(lower::Vector{<:Real}, upper::Vector{<:Real})

Apply a linear stretch to scale all values to the range [0, 1]. The arguments `lower` and
`upper` specify the lower and upper bounds from each channel in the source image. Values that 
either fall below `lower` or above `upper` will be clamped.

# Parameters
- `lower`: The lower-bounds to use for each channel in the source image.
- `upper`: The upper-bounds to use for each channel in the source image.
"""
struct Scale <: AbstractTransform
    lower::Vector{Float64}
    upper::Vector{Float64}
    Scale(lower::Vector{Float64}, upper::Vector{Float64}) = new(lower, upper)
    Scale(lower::Vector{<:Real}, upper::Vector{<:Real}) = Scale(Float64.(lower), Float64.(upper))
end

description(x::Scale) = "Scale values to [0, 1]."

apply(t::Scale, x::AbstractImage, ::Int) = linear_stretch(x, t.lower, t.upper)

"""
    PerImageScale(;lower=0.02, upper=0.98)

Apply a linear stretch to scale all values to the range [0, 1]. The arguments `lower` and
`upper` specify the percentiles at which to define the lower and upper bounds from each channel
in the source image. Values that either fall below `lower` or above `upper` will be clamped.

# Parameters
- `lower`: The quantile to use as the lower-bound in the source array.
- `upper`: The quantile to use as the upper-bound in the source array.
"""
struct PerImageScale{B<:Tuple} <: AbstractTransform
    bounds::B
end

function Scale(;lower=0.02, upper=0.98)
    @argcheck 0 <= lower <= upper <= 1
    return PerImageScale((lower, upper))
end

description(x::PerImageScale) = "Scale values to [0, 1]."

apply(t::PerImageScale, x::AbstractImage, ::Int) = per_image_linear_stretch(x, t.bounds[1], t.bounds[2])

"""
    Normalize(;mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Normalize the channels to have a mean of 0 and a standard deviation of 1.

# Parameters
- `mean`: The channel-wise mean of the input data (uses the ImageNet mean by default).
- `std`: The channel-wise standard deviation of the input data (uses the ImageNet std by default).
"""
struct Normalize <: AbstractTransform
    μ::Vector{Float64}
    σ::Vector{Float64}
end

Normalize(;mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) = Normalize(mean, std)
Normalize(μ::Vector{<:Real}, σ::Vector{<:Real}) = Normalize(Float64.(μ), Float64.(σ))

apply(t::Normalize, x::AbstractImage, ::Int) = normalize(x, t.μ, t.σ)

description(x::Normalize) = "Normalize channels."

"""
    PerImageNormalize()

Normalize the channels to have a mean of 0 and a standard deviation of 1 based on statistics
calculated for each image in a batch.
"""
struct PerImageNormalize <: AbstractTransform end

apply(::PerImageNormalize, x::AbstractImage, ::Int) = per_image_normalize(x)

description(x::PerImageNormalize) = "Normalize with per-image statistics."

"""
    RandomCrop(size::Int)
    RandomCrop(size::Tuple{Int,Int})

Crop a randomly placed tile equal to `size` from the input array.
"""
struct RandomCrop <: AbstractTransform
    sz::Tuple{Int,Int}
end

RandomCrop(size::Int) = RandomCrop((size, size))

apply(t::RandomCrop, x::DType, seed::Int) = random_crop(seed, x, t.sz)

description(x::RandomCrop) = "Random crop to $(x.sz)."

"""
    CenterCrop(size::Int)
    CenterCrop(size::Tuple{Int,Int})

Crop a tile equal to `size` from the center of the input array.
"""
struct CenterCrop <: AbstractTransform
    sz::Tuple{Int,Int}
end

CenterCrop(sz::Int) = CenterCrop((sz,sz))

apply(t::CenterCrop, x::DType, ::Int) = center_crop(x, t.sz)

description(x::CenterCrop) = "Center crop to $(x.sz)."

# FlipX

"""
    FlipX(p)

Apply a random horizontal flip with probability `p`.
"""
struct FlipX <: AbstractTransform
    p::Float64

    FlipX(p::Real) = FlipX(Float64(p))
    function FlipX(p::Float64)
        @argcheck 0 <= p <= 1
        return new(p)
    end
end

apply(t::FlipX, x::DType, seed::Int) = roll_dice(seed, t.p) ? flipX(x) : x

description(x::FlipX) = "Random horizontal flip with probability $(round(x.p, digits=2))."

# FlipY

"""
    FlipY(p)

Apply a random vertical flip with probability `p`.
"""
struct FlipY <: AbstractTransform
    p::Float64

    FlipY(p::Real) = FlipY(Float64(p))
    function FlipY(p::Float64)
        @argcheck 0 <= p <= 1
        return new(p)
    end
end

apply(t::FlipY, x::DType, seed::Int) = roll_dice(seed, t.p) ? flipY(x) : x

description(x::FlipY) = "Random vertical flip with probability $(round(x.p, digits=2))."

# Rot90

"""
    Rot90(p)

Apply a random 90 degree rotation with probability `p`.
"""
struct Rot90 <: AbstractTransform
    p::Float64

    Rot90(p::Real) = Rot90(Float64(p))
    function Rot90(p::Float64)
        (0 <= p <= 1) || throw(ArgumentError("p must be between 0 and 1!"))
        return new(p)
    end
end

apply(t::Rot90, x::DType, seed::Int) = roll_dice(seed, t.p) ? rot90(x) : x

description(x::Rot90) = "Random 90 degree rotation with probability $(round(x.p, digits=2))."

# ColorJitter

"""
    ColorJitter(;contrast=0.5:0.1:1.5, brightness=-0.8:0.1:0.8)

Apply a random color jittering transformations (contrast and brightness adjustments)
according to the formula `α * x + β * M`, where `α` is contrast, `β` is brightness, 
and `M` is either the mean or maximum value of `x`.
"""
struct ColorJitter{C,B} <: AbstractTransform
    contrast::C
    brightness::B
end

function ColorJitter(;contrast::AbstractVector{<:Real}=0.5:0.1:1.5, brightness::AbstractVector{<:Real}=-0.8:0.1:0.8)
    return ColorJitter(contrast, brightness)
end

apply(t::ColorJitter, x::AbstractImage, seed::Int) = color_jitter(MersenneTwister(seed), x, t.contrast, t.brightness)

description(x::ColorJitter) = "Apply random color jitter."

"""
    RandomInvert(p)

Apply a random color inversion with probability `p`.
"""
struct RandomInvert <: AbstractTransform
    p::Float64

    function RandomInvert(p::Float64)
        @argcheck 0 <= p <= 1
        return new(p)
    end
end

apply(t::RandomInvert, x::AbstractImage, seed::Int) = apply_random(invert, seed, t.p, x)

description(x::RandomInvert) = "Invert colors with probability $(x.p)."

"""
    RandomSolarize(p; threshold=0.75)

Randomly solarize an image with probability `p`.
"""
struct RandomSolarize{T} <: AbstractTransform
    p::Float64
    threshold::T

    function RandomSolarize(p::Float64; threshold=0.75)
        @argcheck 0 <= p <= 1
        return new(p, threshold)
    end
end

apply(t::RandomSolarize, x::AbstractImage, seed::Int) = apply_random(x -> solarize(x; t.threshold), seed, t.p, x)

description(x::RandomSolarize) = "Solarize colors with probability $(x.p)."

# Composed Transform

"""
    ComposedTransform(transforms...)

Apply `transforms` to the input in the same order as they are given.

# Example
```julia
julia> r = Raster(rand(256,256, 3), (X,Y,Band));

julia> t = Resample(2.0) |> Tensor();

julia> apply(t, Image(), r, 123) |> size
(512, 512, 3, 1)

julia> apply(t, Image(), r, 123) |> typeof
Array{Float32, 4}
```
"""
struct ComposedTransform{T} <: AbstractTransform
    transforms::T

    function ComposedTransform(transforms::Vararg{AbstractTransform})
        return new{typeof(transforms)}(transforms)
    end
end

function apply(t::ComposedTransform, x::DType, seed::Int)
    seeds = rand(MersenneTwister(seed), 1:10000, length(t.transforms))
    for (s, t) in zip(seeds, t.transforms)
        x = apply(t, x, s)
    end
    return x
end

function Base.show(io::IO, x::ComposedTransform)
    print(io, "$(length(x.transforms))-step ComposedTransform:")
    for (i, t) in enumerate(x.transforms) 
     print(io, "\n  $i) $(description(t))")
    end
end

(|>)(a::AbstractTransform, b::AbstractTransform) = ComposedTransform(a, b)
(|>)(a::ComposedTransform, b::AbstractTransform) = ComposedTransform(a.transforms..., b)
(|>)(a::AbstractTransform, b::ComposedTransform) = ComposedTransform(a, b.transforms...)
(|>)(a::ComposedTransform, b::ComposedTransform) = ComposedTransform(a.transforms..., b.transforms...)
import Base.|>

abstract type AbstractTransform end

apply(::AbstractTransform, x::Any, ::Int) = x

outsize(::AbstractTransform, insize::Tuple) = insize

"""
    transform(t::AbstractTransform, item, x)
    transform(t::AbstractTransform, items::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `item`.
"""
transform(t::AbstractTransform, ::Type{T}, x) where {T <: Item} = apply(t, T(x), rand(1:10000)) |> parent
function transform(t::AbstractTransform, items::Tuple, x::Tuple)
    @argcheck length(items) == length(x)
    seed = rand(1:10000)
    ntuple(length(items)) do i
        apply(t, items[i](x[i]), seed) |> parent
    end
end

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

apply(t::Scale, x::AbstractImage, ::Int) = linear_stretch(x, t.lower, t.upper, channeldim(x))

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

function PerImageScale(;lower=0.02, upper=0.98)
    @argcheck 0 <= lower <= upper <= 1
    return PerImageScale((lower, upper))
end

description(x::PerImageScale) = "Scale values to [0, 1]."

function apply(t::PerImageScale, x::AbstractImage, ::Int)
    modify(data -> per_image_linear_stretch(data, t.bounds[1], t.bounds[2], channeldim(x)), x)
end

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

apply(t::Normalize, x::AbstractImage, ::Int) = normalize(x, t.μ, t.σ; channeldim=channeldim(x))

description(x::Normalize) = "Normalize channels."

"""
    PerImageNormalize()

Normalize the channels to have a mean of 0 and a standard deviation of 1 based on statistics
calculated for each image in a batch.
"""
struct PerImageNormalize <: AbstractTransform end

apply(::PerImageNormalize, x::AbstractImage, ::Int) = per_image_normalize(x; channeldim=channeldim(x))

description(x::PerImageNormalize) = "Normalize with per-image statistics."

# ColorJitter

struct ApplyRandom{T<:AbstractTransform} <: AbstractTransform
    p::Float64
    transform::T
end

function ApplyRandom(t, p::Real)
    @argcheck 0 <= p <= 1 "p must be between 0 and 1!"
    return ApplyRandom(p, t)
end

description(t::ApplyRandom) = "$(description(t.transform)) with probability $(t.p)."

function apply(t::ApplyRandom, x, seed::Int)
    outcome = rand(rng_from_seed(seed), Random.uniform(Float64))
    return outcome <= t.p ? apply(t.transform, x, seed) : x
end

struct OneOf{T} <: AbstractTransform
    transforms::T
end

description(t::OneOf) = join(t.transforms, " or ") * "."

function apply(t::OneOf, x, seed::Int)
    rng = rng_from_seed(seed)
    transform = rand(rng, t.transforms)
    return apply(transform, x, rand(rng,1:10000))
end

struct TrivialAugment <: AbstractTransform
    transforms::Vector{Symbol}
end

description(x::TrivialAugment) = "Apply Trivial Augmentation."

function TrivialAugment(; 
    transforms = [:identity, :rotate, :flipx, :flipy, :zoom, :contrast, :brightness, :sharpen, :blur, :solarize, :grayscale, :permute_channels, :color_jitter] )
    return TrivialAugment(transforms)
end

function apply(t::TrivialAugment, x::Item, seed::Int)
    rng = rng_from_seed(seed)
    _transform = rand(rng, t.transforms)
    _strength = rand(rng, 1:10)
    @match _transform begin
        :identity => x
        :rotate => rot90(x)
        :flipx => flipX(x)
        :flipy => flipY(x)
        :zoom => _random_zoom(seed, x, _strength)
        :contrast => _random_contrast(rng, x, _strength)
        :brightness => _random_brightness(rng, x, _strength)
        :sharpen => _random_sharpen(x, _strength)
        :blur => _random_blur(x, _strength)
        :solarize => _solarize(x, _strength)
        :grayscale => _grayscale(x)
        :permute_channels => _permute_channels(x)
        :color_jitter => _color_jitter(seed, x, _strength)
    end
end

function _random_zoom(seed::Int, x::AbstractMask, strength::Int)
    modify(x -> random_zoom(seed, x, LinRange(1.1, 2, 10)[strength], :nearest), x)
end
function _random_zoom(seed::Int, x::AbstractImage, strength::Int)
    modify(x -> random_zoom(seed, x, LinRange(1.1, 2, 10)[strength], :bilinear), x)
end

_random_contrast(rng, x::AbstractMask, ::Int) = x
function _random_contrast(rng, x::AbstractImage, strength::Int)
    @argcheck 1 <= strength <= 10
    contrast_magnitude = LinRange(0.05, 0.5, 10)[strength]
    contrast = rand(rng, [1 - contrast_magnitude, 1 + contrast_magnitude])
    return adjust_contrast(x, contrast)
end

_random_brightness(rng, x::AbstractMask, ::Int) = x
function _random_brightness(rng, x::AbstractImage, strength::Int)
    @argcheck 1 <= strength <= 10
    brightness_magnitude = LinRange(0.05, 0.30, 10)[strength]
    brightness = rand(rng, [-brightness_magnitude, brightness_magnitude])
    return adjust_brightness(x, brightness)
end

_random_blur(x::AbstractMask, ::Int) = x
_random_blur(x::AbstractImage, strength::Int) = modify(x -> blur(x, strength / 2), x)

_random_sharpen(x::AbstractMask, ::Int) = x
_random_sharpen(x::AbstractImage, strength::Int) = modify(x -> sharpen(x, LinRange(0.1, 0.8, 10)[strength]), x)

_solarize(x::AbstractMask, ::Int) = x
_solarize(x::AbstractImage, strength::Int) = solarize(x; threshold=LinRange(1.0, 0.1, 10)[strength])

_grayscale(x::AbstractMask) = x
_grayscale(x::AbstractImage) = grayscale(x, channeldim(x))

_permute_channels(x::AbstractMask) = x
_permute_channels(x::AbstractImage) = permute_channels(x, channeldim(x))

_color_jitter(::Int, x::AbstractMask, ::Int) = x
_color_jitter(seed::Int, x::AbstractImage, strength::Int) = color_jitter(seed, x, strength, channeldim(x))

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

function apply(t::ComposedTransform, x::Item, seed::Int)
    seeds = rand(rng_from_seed(seed), 1:10000, length(t.transforms))
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
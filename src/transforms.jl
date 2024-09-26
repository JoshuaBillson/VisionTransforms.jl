import Base.|>

struct Image end
struct Mask end
struct NoOp end

struct Image2D{T} <: AbstractArray{T,4}
    data::Array{T,4}
    Image2D(x::AbstractArray{<:Any,4}) = Image2D(Array(x))
    Image2D(x::Array{T,4}) where T = new{T}(x)
end

Base.size(x::Image2D) = size(x.data)

Base.getindex(x::Image2D, i::Int) = x.data[i]

Base.setindex!(x::Image2D, v, i::Int) = Base.setindex!(x.data, v, i)

Base.IndexStyle(::Type{<:Image2D}) = IndexLinear()

Base.similar(x::Image2D, ::Type{T}, dims::Dims) where {T} = Image2D(Base.similar(x.data, T, dims))

struct Mask2D{T} <: AbstractArray{T,4}
    data::Array{T,4}
    Mask2D(x::AbstractArray{<:Any,4}) = Mask2D(Array(x))
    Mask2D(x::Array{T,4}) where T = new{T}(x)
end

Base.size(x::Mask2D) = size(x.data)

Base.getindex(x::Mask2D, i::Int) = x.data[i]

Base.setindex!(x::Mask2D, v, i::Int) = Base.setindex!(x.data, v, i)

Base.IndexStyle(::Type{<:Mask2D}) = IndexLinear()

Base.similar(x::Mask2D, ::Type{T}, dims::Dims) where {T} = Mask2D(Base.similar(x.data, T, dims))

abstract type AbstractTransform end

apply(::AbstractTransform, x, ::Int) = x

"""
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
apply_transform(t::AbstractTransform, x) = apply(t, x, rand(1:1000))
function apply_transform(t::AbstractTransform, x::Tuple)
    seed = rand(1:1000)
    map(x -> apply(t, x, seed), x)
end

"""
    Resize(sz::Tuple)

Resample `x` according to the specified `scale`. `Mask` types will always be
resampled with `:near` interpolation, whereas `Images` will be resampled with 
either `:bilinear` (`scale` > `1`) or `:average` (`scale` < `1`).

# Parameters
- `x`: The raster/stack to be resampled.
- `scale`: The size of the output with respect to the input.
"""
struct Resize{S<:Tuple} <: AbstractTransform
    sz::S
end

apply(t::Resize{Tuple{Int,Int}}, x::Mask2D, ::Int) = imresize(x.data, t.sz, method=:nearest) |> Mask2D
apply(t::Resize{Tuple{Int,Int}}, x::Image2D, ::Int) = imresize(x.data, t.sz, method=:bilinear) |> Image2D

description(x::Resize) = "Resize to $(x.sz)."

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


apply(t::Normalize, x::Image2D, ::Int) = normalize(x.data, t.μ, t.σ; dim=3) |> Image2D

description(x::Normalize) = "Normalize channels."

"""
    RandomCrop(size::Int)
    RandomCrop(size::Tuple{Int,Int})

Crop a randomly placed tile equal to `size` from the input array.
"""
struct RandomCrop <: AbstractTransform
    sz::Tuple{Int,Int}
end

RandomCrop(size::Int) = RandomCrop((size, size))

apply(t::RandomCrop, x::Mask2D, seed::Int) = _apply(t, x, seed)
apply(t::RandomCrop, x::Image2D, seed::Int) = _apply(t, x, seed)
function _apply(t::RandomCrop, x::AbstractArray, seed::Int)
    xpad = size(x, 1) - t.sz[1]
    ypad = size(x, 2) - t.sz[2]
    outcome = rand(Random.MersenneTwister(seed), Random.uniform(Float64), 2)
    ul = max.(ceil.(Int, (xpad, ypad) .* outcome), 1)
    return crop(x, t.sz, ul)
end

description(x::RandomCrop) = "Random crop to $(x.sz)."

# FlipX

"""
    FlipX(p)

Apply a random horizontal flip with probability `p`.
"""
struct FlipX <: AbstractTransform
    p::Float64

    FlipX(p::Real) = FlipX(Float64(p))
    function FlipX(p::Float64)
        (0 <= p <= 1) || throw(ArgumentError("p must be between 0 and 1!"))
        return new(p)
    end
end

apply(t::FlipX, x::Mask2D, seed::Int) = _apply(t, x, seed)
apply(t::FlipX, x::Image2D, seed::Int) = _apply(t, x, seed)
_apply(t::FlipX, x::AbstractArray, seed::Int) = _apply_random(seed, t.p) ? flipX(x) : x

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
        (0 <= p <= 1) || throw(ArgumentError("p must be between 0 and 1!"))
        return new(p)
    end
end

apply(t::FlipY, x::Mask2D, seed::Int) = _apply(t, x, seed)
apply(t::FlipY, x::Image2D, seed::Int) = _apply(t, x, seed)

_apply(t::FlipY, x::AbstractArray, seed::Int) = _apply_random(seed, t.p) ? flipY(x) : x

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

apply(t::Rot90, x::Mask2D, seed::Int) = _apply(t, x, seed)
apply(t::Rot90, x::Image2D, seed::Int) = _apply(t, x, seed)

_apply(t::Rot90, x::AbstractArray, seed::Int) = _apply_random(seed, t.p) ? rot90(x) : x

description(x::Rot90) = "Random 90 degree rotation with probability $(round(x.p, digits=2))."

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

function apply(t::ComposedTransform, x, seed::Int)
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

function _apply_random(seed::Int, p::Float64)
    @assert 0 <= p <= 1 "p must be between 0 and 1!"
    outcome = rand(Random.MersenneTwister(seed), Random.uniform(Float64))
    return outcome <= p
end
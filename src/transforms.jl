import Base.|>

struct Image end
struct Mask end
struct NoOp end

abstract type AbstractTransform end

apply(::AbstractTransform, ::Any, x, ::Int) = x

"""
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
apply_transform(t::AbstractTransform, dtype, data) = apply(t, dtype, data, rand(1:1000))
function apply_transform(t::AbstractTransform, dtype::Tuple, data::Tuple)
    @assert length(dtype) == length(data)
    seed = rand(1:1000)
    return ntuple(i -> apply(t, dtype[i], data[i], seed), length(data))
end

"""
    Resample(scale)

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

apply(t::Resize, ::Mask, x, ::Int) = imresize(x, t.sz, method=:nearest)
apply(t::Resize, ::Image, x, ::Int) = imresize(x, t.sz, method=:bilinear)

description(x::Resize) = "Resize to $(x.sz)."
"""
    RandomCrop(size::Int)
    RandomCrop(size::Tuple{Int,Int})

Crop a randomly placed tile equal to `size` from the input array.
"""
struct RandomCrop <: AbstractTransform
    sz::Tuple{Int,Int}
end

RandomCrop(size::Int) = RandomCrop((size, size))

function apply(t::RandomCrop, ::Union{Image,Mask}, x, seed::Int)
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

apply(t::FlipX, ::Union{Mask,Image}, x, seed::Int) = _apply_random(seed, t.p) ? flipX(x) : x

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

apply(t::FlipY, ::Union{Image,Mask}, x, seed::Int) = _apply_random(seed, t.p) ? flipY(x) : x

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

apply(t::Rot90, ::Union{Image,Mask}, x, seed::Int) = _apply_random(seed, t.p) ? rot90(x) : x

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

function apply(t::ComposedTransform, dtype, x, seed::Int)
    return reduce((acc, trans) -> apply(trans, dtype, acc, seed), t.transforms, init=x)
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
import Base.|>

abstract type DType{T,N} <: AbstractArray{T,N} end

abstract type AbstractImage{T,N} <: DType{T,N} end

abstract type AbstractMask{T,N} <: DType{T,N} end

struct Image2D{T} <: AbstractImage{T,4}
    data::Array{T,4}
    Image2D(x::AbstractArray{<:Any,4}) = Image2D(Array(x))
    Image2D(x::AbstractArray{<:Images.Colorant,2}) = Image2D(image2tensor(x))
    Image2D(x::Array{T,4}) where T = new{T}(x)
end

struct Image3D{T} <: AbstractImage{T,5}
    data::Array{T,5}
    Image3D(x::AbstractArray{<:Any,5}) = Image3D(Array(x))
    Image3D(x::Array{T,5}) where T = new{T}(x)
end

struct Series2D{T} <: AbstractImage{T,5}
    data::Array{T,5}
    Series2D(x::AbstractArray{<:Any,5}) = Series2D(Array(x))
    Series2D(x::Array{T,5}) where T = new{T}(x)
    function Series2D(x::AbstractVector{<:AbstractArray{<:Images.Colorant,2}})
        @pipe map(image2tensor, x) |> map(x -> unsqueeze(x, 4), _) |> cat(_..., dims=4)
    end
end

struct Mask2D{T} <: AbstractMask{T,4}
    data::Array{T,4}
    Mask2D(x::AbstractArray{<:Any,4}) = Mask2D(Array(x))
    Mask2D(x::Array{T,4}) where T = new{T}(x)
end

struct Mask3D{T} <: AbstractMask{T,5}
    data::Array{T,5}
    Mask3D(x::AbstractArray{<:Any,5}) = Mask3D(Array(x))
    Mask3D(x::Array{T,5}) where T = new{T}(x)
end

struct NoOp{T,N} <: DType{T,N}
    data::Array{T,N}
    NoOp(x::AbstractArray) = Mask3D(Array(x))
    NoOp(x::Array{T,N}) where {T,N} = new{T,N}(x)
end

for dtype = (:Image2D, :Image3D, :Series2D, :Mask2D, :Mask3D, :NoOp)
    @eval Base.size(x::$dtype) = size(x.data)

    @eval Base.getindex(x::$dtype, i::Int) = x.data[i]

    @eval Base.setindex!(x::$dtype, v, i::Int) = Base.setindex!(x.data, v, i)

    @eval Base.IndexStyle(::Type{<:$dtype}) = IndexLinear()

    @eval Base.similar(x::$dtype, ::Type{T}, dims::Dims) where {T} = $dtype(Base.similar(x.data, T, dims))

    @eval Base.BroadcastStyle(::Type{<:$dtype}) = Broadcast.ArrayStyle{$dtype}()

    @eval begin 
        function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{$dtype}}, ::Type{T}) where T
            return $dtype(similar(Array{T}, axes(bc)))
        end
    end
end

abstract type AbstractTransform end

apply(::AbstractTransform, x::DType, ::Int) = x
apply(::AbstractTransform, x::NoOp, ::Int) = x
function apply(::AbstractTransform, x::T, ::Int) where T
    throw(ArgumentError("Transform expects input to sub-type `DType`, but received $T."))
end

"""
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
transform(t::AbstractTransform, ::Type{T}, x) where {T <: DType} = transform(t, T(x))
transform(t::AbstractTransform, x) = apply(t, x, rand(1:1000))
function transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)
    return transform(t, ntuple(i -> dtypes[i](x[i]), length(dtypes)))
end
function transform(t::AbstractTransform, x::Tuple)
    seed = rand(1:1000)
    map(x -> apply(t, x, seed), x)
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

apply(t::Resize{Tuple{Int,Int}}, x::Mask2D, ::Int) = imresize(x, t.sz, method=:nearest)
apply(t::Resize{Tuple{Int,Int}}, x::Image2D, ::Int) = imresize(x, t.sz, method=:bilinear)
apply(t::Resize{Tuple{Int,Int}}, x::Series2D, ::Int) = imresize(x, t.sz, method=:bilinear)
apply(t::Resize{Tuple{Int,Int,Int}}, x::Image3D, ::Int) = imresize(x, t.sz, method=:bilinear)
apply(t::Resize{Tuple{Int,Int,Int}}, x::Mask3D, ::Int) = imresize(x, t.sz, method=:nearest)

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

apply(t::Scale, x::Image2D, ::Int) = linear_stretch(x, t.lower, t.upper, 3)
apply(t::Scale, x::Image3D, ::Int) = linear_stretch(x, t.lower, t.upper, 4)
apply(t::Scale, x::Series2D, ::Int) = linear_stretch(x, t.lower, t.upper, 4)

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

apply(t::PerImageScale, x::Image2D, ::Int) = per_image_linear_stretch(x, t.bounds[1], t.bounds[2], 3)
apply(t::PerImageScale, x::Image3D, ::Int) = per_image_linear_stretch(x, t.bounds[1], t.bounds[2], 4)
apply(t::PerImageScale, x::Series2D, ::Int) = per_image_linear_stretch(x, t.bounds[1], t.bounds[2], 4)

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

apply(t::Normalize, x::Image2D, ::Int) = normalize(x.data, t.μ, t.σ; dim=3)
apply(t::Normalize, x::Image3D, ::Int) = normalize(x.data, t.μ, t.σ; dim=4)
apply(t::Normalize, x::Series2D, ::Int) = normalize(x.data, t.μ, t.σ; dim=3)

description(x::Normalize) = "Normalize channels."

"""
    PerImageNormalize()

Normalize the channels to have a mean of 0 and a standard deviation of 1 based on statistics
calculated for each image in a batch.
"""
struct PerImageNormalize <: AbstractTransform end

apply(::PerImageNormalize, x::Image2D, ::Int) = per_image_normalize(x, 3)
apply(::PerImageNormalize, x::Image3D, ::Int) = per_image_normalize(x, 4)
apply(::PerImageNormalize, x::Series2D, ::Int) = per_image_normalize(x, 3)

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

apply(::RandomCrop, x::NoOp, ::Int) = x
apply(t::RandomCrop, x::DType, seed::Int) = _apply(t, x, seed)
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

apply(::FlipX, x::NoOp, ::Int) = x
apply(t::FlipX, x::DType, seed::Int) = _apply_random(seed, t.p) ? flipX(x) : x

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

apply(::FlipY, x::NoOp, ::Int) = x
apply(t::FlipY, x::DType, seed::Int) = _apply_random(seed, t.p) ? flipY(x) : x

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

apply(::Rot90, x::NoOp, ::Int) = x
apply(t::Rot90, x::DType, seed::Int) = _apply_random(seed, t.p) ? rot90(x) : x

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
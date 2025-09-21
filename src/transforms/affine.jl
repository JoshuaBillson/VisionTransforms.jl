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

apply(::Resize, x, ::Int) = x
apply(t::Resize, x::AbstractMask, ::Int) = imresize(x, t.sz, :nearest)
apply(t::Resize, x::AbstractImage, ::Int) = imresize(x, t.sz, :bilinear)

description(x::Resize) = "Resize to $(x.sz)."

"""
    Crop(from::CropFrom, sz::Tuple)

Crop a raster of size `sz` from the location specified by `from`.
"""
struct Crop{F<:CropFrom,T<:Tuple} <: AbstractTransform
    from::F
    sz::T

    function Crop(from::F, sz::T) where {F<:CropFrom,T<:Tuple}
        @argcheck all(sz .>= 1)
        new{F,T}(from, sz)
    end
end

description(t::Crop{FromCenter}) = "Center crop to size $(t.sz)."
description(t::Crop{FromOrigin}) = "Origin crop to size $(t.sz)."
description(t::Crop{FromRandom}) = "Random crop to size $(t.sz)."

function apply(t::Crop, x::AbstractRaster, seed::Int)
    sz = _crop_size(imsize(x), t.sz)  # Adjust crop dimensions to match image dimensions
    return crop(rng_from_seed(seed), t.from, sz, x)
end

_crop_size(imsize::NTuple{N,Int}, cropsize::NTuple{N,Int}) where N = cropsize
function _crop_size(imsize::NTuple{N1,Int}, cropsize::NTuple{N2,Int}) where {N1,N2}
    if N1 < N2  # imsize has fewer dimensions than cropsize
        return cropsize[1:N1]
    else  # imsize has more dimensions than cropsize
        return ntuple(i -> i <= N2 ? cropsize[i] : imsize[i], N1)
    end
end

"""
    RandomCrop(sz)

Crops a randomly placed tile of size `sz`.
"""
RandomCrop(sz) = Crop(FromRandom(), sz)

"""
    OriginCrop(sz)

Crops a tile of size `sz` from the origin.
"""
OriginCrop(sz) = Crop(FromOrigin(), sz)

"""
    CenterCrop(sz)

Crops a tile of size `sz` from the center.
"""
CenterCrop(sz) = Crop(FromCenter(), sz)

"""
    Zoom(from::CropFrom; strength=1.0:0.1:2.0)

Zoom to the location specified by `from` by a factor sampled from `strength`.
"""
struct Zoom{F<:CropFrom,S} <: AbstractTransform
    from::F
    strength::S
end

function Zoom(from::CropFrom; strength=1.0:0.1:2.0)
    @argcheck all(x -> x >= 1, strength)
    return Zoom(from, strength)
end

description(t::Zoom{FromCenter}) = "Center zoom by a factor sampled from $(t.strength)."
description(t::Zoom{FromOrigin}) = "Origin zoom by a factor sampled from $(t.strength)."
description(t::Zoom{FromRandom}) = "Random zoom by a factor sampled from $(t.strength)."

function apply(t::Zoom, x::AbstractImage, seed::Int)
    rng = rng_from_seed(seed)
    zoom_strength = rand(rng, t.strength)
    return zoom(rng, t.from, zoom_strength, :bilinear, x)
end

function apply(t::Zoom, x::AbstractMask, seed::Int)
    rng = rng_from_seed(seed)
    zoom_strength = rand(rng, t.strength)
    return zoom(rng, t.from, zoom_strength, :nearest, x)
end

"""
    RandomZoom(;strength=1.0:0.1:2.0)

Zoom to a random point by a factor sampled from `strength`.
"""
RandomZoom(;strength=1.0:0.1:2.0) = Zoom(FromRandom(), strength)

"""
    CenterZoom(;strength=1.0:0.1:2.0)

Zoom to the center of a mask/image by a factor sampled from `strength`.
"""
CenterZoom(;strength=1.0:0.1:2.0) = Zoom(FromCenter(), strength)

"""
    Flip(dim::Int; p=1.0)

Flip the input raster along the specified dimension with probability `p`.
"""
struct Flip <: AbstractTransform
    p::Float64
    dim::Int
end

function Flip(dim::Int; p=1.0)
    @argcheck dim > 0
    @argcheck 0 <= p <= 1
    return Flip(p, dim)
end

function description(t::Flip)
    @match t.dim begin
        1 => "Flip across Y axis with probability $(t.p)."
        2 => "Flip across X axis with probability $(t.p)."
        3 => "Flip across Z axis with probability $(t.p)."
        x => "Flip dimension $x with probability $(t.p)."
    end
end

function apply(t::Flip, x::AbstractRaster{<:Any,N}, seed::Int) where N
    return roll_dice(rng_from_seed(seed), t.p) ? flip(x, t.dim) : x
end

"""
    FlipX(;p=1.0)

Apply a random horizontal flip with probability `p`.
"""
FlipX(;p=1.0) = Flip(2; p)

"""
    FlipY(;p=1.0)

Apply a random vertical flip with probability `p`.
"""
FlipY(;p=1.0) = Flip(1; p)

"""
    FlipZ(;p=1.0)

Apply a random depth flip with probability `p`.
"""
FlipZ(;p=1.0) = Flip(3; p)

"""
    Rot90(;p=1.0)

Apply a random 90 degree rotation with probability `p`.
"""
struct Rot90 <: AbstractTransform
    p::Float64
end

function Rot90(;p=1.0)
    @argcheck 0 <= p <= 1
    return Rot90(p)
end

apply(t::Rot90, x::Item, seed::Int) = roll_dice(rng_from_seed(seed), t.p) ? rot90(x) : x

description(x::Rot90) = "Random 90 degree rotation with probability $(round(x.p, digits=2))."

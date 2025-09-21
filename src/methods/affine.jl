abstract type CropFrom end

struct FromCenter <: CropFrom end

struct FromOrigin <: CropFrom end

struct FromRandom <: CropFrom end

sample_tile(c::FromCenter, args...) = sample_tile(Random.default_rng(), c, args...)
function sample_tile(::Random.AbstractRNG, ::FromCenter, imgsize::NTuple{N,Int}, tilesize::NTuple{N,Int}) where N
    return ((imgsize .- tilesize) .÷ 2) .+ 1
end

sample_tile(c::FromOrigin, args...) = sample_tile(Random.default_rng(), c, args...)
function sample_tile(::Random.AbstractRNG, ::FromOrigin, imgsize::NTuple{N,Int}, tilesize::NTuple{N,Int}) where N
    return ntuple(_ -> 1, N)
end

sample_tile(c::FromRandom, args...) = sample_tile(Random.default_rng(), c, args...)
function sample_tile(rng::Random.AbstractRNG, ::FromRandom, imgsize::NTuple{N,Int}, tilesize::NTuple{N,Int}) where N
    # Get Upper and Lower Bounds
    lower_bounds = ntuple(_ -> 1, N)
    upper_bounds = imgsize .- tilesize .+ 1

    # Sample Random Point From Bounds
    map((1:N...,)) do i
        outcome = rand(rng, Random.uniform(Float64))
        displacement = round(Int, outcome * (upper_bounds[i] - lower_bounds[i]))
        return lower_bounds[i] + displacement
    end
end

"""
    crop([rng], from::CropFrom, sz::NTuple{N1,Int}, x::AbstractArray{<:Any,N2})

Crop `x` to `sz` with the upper-left corner specified by `from`.
"""
crop(from::CropFrom, sz::NTuple, x::AbstractArray) = crop(Random.default_rng(), from, sz, x)
function crop(rng::Random.AbstractRNG, from::CropFrom, sz::NTuple{N1,Int}, x::AbstractArray{<:Any,N2}) where {N1,N2}
    ul = sample_tile(rng, from, imsize(x), sz)
    return _crop(x, sz, ul)
end

function _crop(x::AbstractArray{<:Any,N1}, sz::NTuple{N2,Int}, ul::NTuple{N2,Int}) where {N1,N2}
    # Arg Checks
    @argcheck N1 == N2 + 1
    @argcheck all(1 .<= sz .<= imsize(x))
    @argcheck all(1 .<= ul .<= imsize(x))
    @argcheck all((ul .+ sz .- 1) .<= imsize(x))

    # Compute Lower-Right Coordinates
    lr = ul .+ sz .- 1

    # Crop Tile
    indices = ntuple(i -> i <= N2 ? (ul[i]:lr[i]) : Colon(), N1)
    return x[indices...]
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
    return mapslices(x -> _imresize(x, sz, method), img; dims=(1:N2...,))
end

function _imresize(img::AbstractArray, sz::Tuple, method::Symbol)
    @match method begin
        :nearest => ImageTransformations.imresize(img, sz, method=Constant())
        :bilinear => ImageTransformations.imresize(img, sz, method=Linear())
    end
end

zoom(from::CropFrom, zoom_strength::Real, method::Symbol, x::AbstractArray) = zoom(Random.default_rng(), from, zoom_strength, method, x)
function zoom(rng::Random.AbstractRNG, from::CropFrom, zoom_strength::Real, method::Symbol, x::AbstractArray)
    @argcheck zoom_strength >= 1
    newsize = Int.(imsize(x) .÷ zoom_strength)
    cropped = crop(rng, from, newsize, x)
    return imresize(cropped, imsize(x), method)
end

function flip(x::AbstractArray{<:Any,N}, dim::Int) where N
    @argcheck dim <= N
    indices = ntuple(i -> i == dim ? (size(x,dim):-1:1) : Colon(), N)
    return x[indices...]
end

"""
    rot90(x)

Rotate the image `x` by 90 degress. 
"""
rot90(x::AbstractArray{<:Any,N}) where N = flip(permutedims(x, (2,1,3:N...)), 2)
function image2tensor(image::AbstractArray{<:Images.Colorant,2})
    @pipe image |>
    Images.float32.(_) |>
    Images.channelview(_) |>
    permutedims(_, (3,2,1)) |>
    _putobs(_)
end

function tensor2image(tensor::AbstractArray{<:Real,4})
    @assert size(tensor, 3) == 3
    if size(tensor, 4) > 1
        return map(i -> tensor2image(selectdim(tensor, 4, i:i)), axes(tensor)[end])
    end
    return @pipe selectdim(tensor, 4, 1) |> Images.n0f8.(_) |> permutedims(_, (3,2,1)) |> Images.colorview(Images.RGB, _)
end

_putobs(x::AbstractArray) = reshape(x, size(x)..., 1)

"""
    resize(img::AbstractArray, sz::Tuple{Int,Int}; method=:bilinear)

Resize `img` to `sz` with the specified resampling `method`.

# Parameters
- `img`: The image to be resized.
- `sz`: The width and height of the output as a tuple.
- `method`: Either `:nearest` or `:bilinear`.
"""
function imresize(img::AbstractArray{<:Real,N}, sz::Tuple; method=:bilinear) where {N}
    if size(img,N) == 1
        resized = _imresize(selectdim(img, N, 1), sz, method)
        return reshape(resized, size(resized)..., 1)
    end
    return _imresize(img, sz, method)
end

function _imresize(img::AbstractArray, sz::Tuple, method::Symbol)
    @match method begin
        :nearest => Images.imresize(img, sz, method=Constant())
        :bilinear => Images.imresize(img, sz, method=Linear())
    end
end

"""
    crop(x, sz::Int, ul=(1,1))
    crop(x, sz::Tuple{Int,Int}, ul=(1,1))

Crop a tile equal to `size` out of `x` with an upper-left corner defined by `ul`.
"""
crop(x::AbstractArray, sz::Int, ul=(1,1)) = crop(x, (sz, sz), ul)
function crop(x::AbstractArray, sz::Tuple{Int,Int}, ul=(1,1))
    # Compute Lower-Right Coordinates
    lr = ul .+ sz .- 1

    # Check Bounds
    any(sz .< 1) && throw(ArgumentError("Crop size must be positive!"))
    (any(ul .< 1) || any(lr .> _size(x))) && throw(ArgumentError("Crop is out of bounds!"))

    # Crop Tile
    return _crop(x, ul[1]:lr[1], ul[2]:lr[2])
end

_size(x::AbstractArray) = size(x)[1:2]

_crop(x::AbstractArray{<:Any,2}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims]
_crop(x::AbstractArray{<:Any,3}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:]
_crop(x::AbstractArray{<:Any,4}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:]
_crop(x::AbstractArray{<:Any,5}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:,:]
_crop(x::AbstractArray{<:Any,6}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:,:,:]

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
function rot90(x::AbstractArray{<:Any,4})
    return @pipe permutedims(x, (2, 1, 3, 4)) |> reverse(_, dims=2)
end
"""
    grayscale(x::AbstractArray, [channeldim])

Convert `x` to a grayscale image.
"""
function grayscale(x::AbstractArray{<:Real,N}, channeldim=N) where N
    repeatdims = ntuple(i -> i == channeldim ? size(x,channeldim) : 1, N)
    return repeat(mean(x, dims=channeldim), repeatdims...)
end

"""
    adjust_contrast(x::AbstractArray, contrast::Real, channeldim::Int)

Adjust the contrast of `x` by `contrast`.
"""
function adjust_contrast(x::AbstractArray{T,N}, contrast::Real) where {T<:Real,N}
    @argcheck 0 < contrast
    return scaled_map(x) do x
        c = T(contrast)
        μ = mean(x, dims=(1:N-1...,))
        return ((x .- μ) .* c) .+ μ
    end
end

"""
    adjust_brightness(x::AbstractArray, brightness::Real)

Adjust the brightness of `x` by `brightness`.
"""
function adjust_brightness(x::AbstractArray{T}, brightness::Real) where {T <: Real}
    @argcheck -1 <= brightness <= 1
    return scaled_map(x -> x .+ T(brightness), x)
end

"""
    shift_hue(x::AbstractArray, shift::Real, [channeldim])

Shift the hue of `x` in the HSV color space by the amount specified by `shift`.
"""
function shift_hue(x::AbstractArray{T,N}, shift::Real, channeldim=N) where {T<:Real,N}
    @argcheck size(x, channeldim) == 3
    return scaled_map(x) do x
        hsv = rgb_to_hsv(x)
        shifted_hsv = (hsv .+ vec2array([shift,0,0], hsv, channeldim)) .% T(360)
        return hsv_to_rgb(shifted_hsv)
    end
end

"""
    color_jitter(rng, x::AbstractArray, strength::Int, [channeldim])

Applies random color jittering transformations (hue, saturation, and brightness) to 
the input image `x`.

# Parameters
- `rng`: A random number generator to make the outcome reproducible.
- `x`: A tensor containing an RGB image or image series.
- `strength`: The strength of the jittering, from 1 to 10.
- `channeldim`: The dimension corresponding to channels in `x`.
"""
function color_jitter(rng, x::AbstractArray{T,3}, strength::Int) where {T<:Real}
    @argcheck 1 <= strength <= 10

    # Get Hue, Value, and Saturation Shift
    hue_shift = LinRange(20,60,10)[strength] * rand(rng, -1:0.1:1)
    saturation_shift = LinRange(0.05,0.40,10)[strength] * rand(rng, -1:0.1:1)
    value_shift = LinRange(0.05,0.40,10)[strength] * rand(rng, -1:0.1:1)
    shift = reshape(T.([hue_shift,saturation_shift,value_shift]), (3,1,1))

    # Apply Shift
    hsv = tensor2image(x) .|> ImageCore.HSV |> ImageCore.channelview
    return hsv
    hsv_shifted = hsv .+ shift
    hsv_shifted[1,:,:] .= hsv_shifted[1,:,:] .% T(360)
    clamp01!((@view hsv_shifted[2:3,:,:]))
    return ImageCore.colorview(ImageCore.HSV, hsv_shifted) .|> ImageCore.RGB |> image2tensor
end

"""
    permute_channels(rng, x::AbstractArray, [channeldim])

Permute the channel ordering of `x`.
"""
function permute_channels(rng, x::AbstractArray{T,N}, channeldim=N) where {T<:Real,N}
    @argcheck 1 <= channeldim <= N
    nchannels = size(x,channeldim)
    indices = ntuple(i -> i == channeldim ? randperm(rng, nchannels) : Colon(), N)
    return x[indices...]
end

"""
    invert_color(x::AbstractArray{<:Real})

Invert the colors of `x`.
"""
function invert_color(x::AbstractArray{T}) where {T <: Real}
    lb = minimum(x)
    ub = maximum(x)
    midpoint = (lb + ub) / T(2)
    return T(2 * midpoint) .- x  # (midpoint - x) + midpoint = 2 * midpoint - x
end

"""
    solarize(x::AbstractArray{<:Real}; threshold=0.75)

Solarize `x` by inverting all values above `threshold`.
"""
function solarize(x::AbstractArray{<:Real}; threshold=0.75)
    lb = minimum(x)
    ub = maximum(x)
    thresh = ((ub - lb) * threshold) + lb
    return ifelse.(x .> thresh, invert_color(x), x)
end
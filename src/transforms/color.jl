"""
    Grayscale(;p=1.0)

Convert to grayscale by averaging the channel dimension with probability `p`.
"""
struct Grayscale <: AbstractTransform
    p::Float64
end

function Grayscale(;p=1.0)
    @argcheck 0 <= p <= 1
    return Grayscale(p)
end

function apply(t::Grayscale, x::AbstractImage, seed::Int)
    @debug "Grayscale"
    return roll_dice(rng_from_seed(seed), t.p) ? grayscale(x) : x
end

description(t::Grayscale) = "Convert to grayscale with probability $(t.p)."

"""
    ColorJitter(;strength=5)

Apply a random color jitter consisting of contrast, brightness, and hue adjustments.
Use `strength` to determine the maximum extent of the jittering, with a value in the
range [1,10].
"""
struct ColorJitter <: AbstractTransform
    strength::Int
end

function ColorJitter(;strength=5)
    return ColorJitter(strength)
end

function apply(t::ColorJitter, x::AbstractImage, seed::Int)
    @debug "Color Jitter"
    return color_jitter(rng_from_seed(seed), x, t.strength)
end

description(x::ColorJitter) = "Apply random color jitter."

"""
    InvertColor(;p=1.0)

Apply a random color inversion with probability `p`.
"""
struct InvertColor <: AbstractTransform
    p::Float64
end

function InvertColor(;p=1.0)
    @argcheck 0 <= p <= 1
    return InvertColor(p)
end

function apply(t::InvertColor, x::AbstractImage, seed::Int) 
    @debug "Invert Color"
    rng = rng_from_seed(seed)
    return roll_dice(rng, t.p) ? invert_color(x) : x
end

description(t::InvertColor) = "Invert colors with probability $(t.p)."

"""
    Solarize(;p=1.0, threshold=0.75)

Randomly solarize an image with probability `p`.
"""
struct Solarize <: AbstractTransform
    p::Float64
    threshold::Float64
end

function Solarize(;p=1.0, threshold=0.75)
    @argcheck 0 <= p <= 1
    @argcheck 0 <= threshold <= 1
    return Solarize(p, threshold)
end

function apply(t::Solarize, x::AbstractImage, seed::Int)
    @debug "Solarize"
    rng = rng_from_seed(seed)
    return roll_dice(rng, t.p) ? solarize(x; t.threshold) : x
end

description(x::Solarize) = "Solarize colors with probability $(x.p)."

struct PermuteChannels <: AbstractTransform
    p::Float64
end

function PermuteChannels(;p=1.0)
    @argcheck 0 <= p <= 1
    return PermuteChannels(p)
end

function apply(t::PermuteChannels, x::AbstractImage, seed::Int)
    @debug "Permute Channels"
    rng = rng_from_seed(seed)
    return roll_dice(rng, t.p) ? permute_channels(rng, x) : x
end

description(t::PermuteChannels) = "Randomly permute channels with probability $(t.p)."
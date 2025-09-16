abstract type AbstractNormalizer end

struct PerImageNormalization <: AbstractNormalizer end

struct FixedNormalization{M,S} <: AbstractNormalizer
    mu::M
    sigma::S
end

struct Recipe{N<:AbstractNormalizer,A}
    aug::A
    normalization::N
end

(r::Recipe)(x::Any, seed=rand(1:10000)) = x
(r::Recipe)(x::Tuple, seed=rand(1:1000)) = map(x->r(x,seed), x)
(r::Recipe)(x::AbstractMask, seed=rand(1:10000)) = apply(r.aug, x, seed)
function (r::Recipe{PerImageNormalization})(x::AbstractImage{T,N}, seed=rand(1:10000)) where {T,N}
    # Perform Linear Stretch
    lb = minimum(x, dims=(1:N-1...,))
    ub = maximum(x, dims=(1:N-1...,))
    scaled_x = clamp!((x .- lb) ./ (ub .- lb), 0, 1)

    # Apply Augmentation
    augmented_scaled_x = apply(r.aug, scaled_x, seed)

    # Normalize
    return _normalize_img(augmented_scaled_x, r.normalization, lb, ub)
end

function _normalize_img(x::AbstractArray{T,N}, norm::FixedNormalization, lb, ub) where {T,N}
    @assert size(x,N) == length(norm.mu) == length(norm.sigma)
    return mapslices((x .* (ub .- lb)) .+ lb, dims=(1:N-1...)) do x
        return (x .- T.(norm.mu)) ./ T.(norm.sigma)
    end
end

function _normalize_img(x::AbstractArray{T,N}, ::PerImageNormalization, lb, ub) where {T,N}
    μ = mean(x, dims=(1:N-1...))
    σ = std(x, dims=(1:N-1...))
    return (x .- μ) ./ σ
end
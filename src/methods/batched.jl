MixUp(images::AbstractArray, labels::AbstractArray; kw...) = MixUp(Random.default_rng(), images, labels; kw...)
function MixUp(rng, images::AbstractArray{T,4}, labels::AbstractArray{<:Real,2}; alpha=0.2) where {T}
    @argcheck size(images,4) > 1
    @argcheck size(images,4) == size(labels,2)

    # Get Paired Images and Labels
    nobs = size(images,4)
    pairs = Random.randperm(rng, nobs)
    images2 = @view images[:,:,:,pairs]
    labels2 = @view labels[:,pairs]

    # Get Mixing Values
    lambdas = T.(Random.rand(rng, Distributions.Beta(alpha), nobs))

    # Mix Images and Labels
    mixed_images = (reshape(lambdas, (1,1,1,:)) .* images) .+ (reshape(T(1) .- lambdas, (1,1,1,:)) .* images2)
    mixed_labels = (reshape(lambdas, (1,:)) .* labels) .+ (reshape(T(1) .- lambdas, (1,:)) .* labels2)
    return mixed_images, mixed_labels
end
MixUp(images::AbstractArray, labels::AbstractArray; kw...) = MixUp(Random.default_rng(), images, labels; kw...)
function MixUp(rng, images::AbstractArray{T,4}, labels::AbstractArray{<:Real,2}; alpha=0.2) where {T}
    @argcheck size(images,4) > 1
    @argcheck size(images,4) == size(labels,2)

    # Get Paired Images and Labels
    nobs = size(images,4)
    pairs = Random.randperm(rng, nobs)

    # Get Mixing Values
    lambdas = T.(Random.rand(rng, Distributions.Beta(alpha), nobs))

    # Mix Images and Labels
    return _mixup(images, labels, pairs, lambdas)
end

function _mixup(images::AbstractArray{T,4}, labels::AbstractArray{T,2}, pairs::AbstractVector{<:Integer}, lambdas::AbstractVector{T}) where {T<:Real}
    # Make pairs and lambdas match type of image and label arrays
    _pairs = similar(images, Int, size(pairs))
    _pairs .= pairs
    _lambdas = similar(images, T, (1,1,1,length(lambdas)))
    _lambdas[1,1,1,:] .= lambdas

    # Mix Images and Labels
    images2 = @view images[:,:,:,_pairs]
    labels2 = @view labels[:,_pairs]
    mixed_images = (_lambdas .* images) .+ ((T(1) .- _lambdas) .* images2)
    mixed_labels = (dropdims(_lambdas; dims=(1,2)) .* labels) .+ ((T(1) .- dropdims(_lambdas; dims=(1,2))) .* labels2)
    return mixed_images, mixed_labels
end
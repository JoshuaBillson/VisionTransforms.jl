vec2array(x::AbstractVector, to::AbstractArray{T}, dim::Int) where T = vec2array(T.(x), to, dim)
function vec2array(x::AbstractVector{T}, to::AbstractArray{T,N}, dim::Int) where {T,N}
    @argcheck 0 < dim <= N
    @argcheck size(to, dim) == length(x)
    newshape = ntuple(i -> i == dim ? length(x) : 1, N)
    return reshape(x, newshape)
end

function unsqueeze(x::AbstractArray{<:Any,N}, dim::Int) where N
    newsize = (size(x)[1:dim-1]..., 1, size(x)[dim:end]...)
    return reshape(x, newsize)
end

function _quantiles(x::AbstractArray{<:Real}, lower, upper, dim::Int)
    map(1:size(x, dim)) do i
        data = selectdim(x, dim, i) |> vec
        return _quantiles(data, lower, upper)
    end
end

function _quantiles(x::AbstractVector{<:Real}, lower, upper)
	# Build Histogram
	edges, counts = @pipe build_histogram(x, 10000)

	# Get Percentiles
	ps = cumsum(counts) ./ sum(counts)

	# Get Lower and Upper Bounds
	lb = @pipe findfirst(>=(lower), ps) |> clamp(_, 1, 10000) |> edges[_]
	ub = @pipe findfirst(>=(upper), ps) |> clamp(_, 1, 10000) |> edges[_]
	return (lb, ub)
end

function build_histogram(img, nbins)
	minval = minimum(img)
	maxval = maximum(img)
    edges = _partition_interval(nbins, minval, maxval)
    return _build_histogram(img, edges)
end

function _build_histogram(img, edges::AbstractRange)
    lb = first(axes(edges,1))
    ub = last(axes(edges,1))
    first_edge, last_edge = first(edges), last(edges)
    inv_step_size = 1/step(edges)
    counts = fill(0, lb:ub)
    @inbounds for val in img
		if isnan(val) || ismissing(val) || isnothing(val)
			continue
		elseif val >= last_edge
            counts[ub] += 1
        else
            index = floor(Int, ((val-first_edge)*inv_step_size)) + 1
            counts[index] += 1
        end
    end
    edges, counts
end

function _partition_interval(nbins::Integer, minval::Real, maxval::Real)
    return range(minval, step=(maxval - minval) / nbins, length=nbins)
end

function apply_random(f, seed::Int, p::Float64, x)
    return roll_dice(seed, p) ? f(x) : x
end

function roll_dice(seed::Int, p::Float64)
    @assert 0 <= p <= 1 "p must be between 0 and 1!"
    outcome = rand(Random.MersenneTwister(seed), Random.uniform(Float64))
    return outcome <= p
end

function random_point(seed::Int, lower_bounds::Tuple, upper_bounds::Tuple)
    @argcheck length(lower_bounds) == length(upper_bounds)
    @argcheck all(lower_bounds .< upper_bounds)
    rng = Random.MersenneTwister(seed)
    dims = length(lower_bounds)  # dimension of the space to be sampled
    span = upper_bounds .- lower_bounds
    outcome = ntuple(_ -> rand(rng, Random.uniform(Float64)), dims)
    displacement = round.(Int, outcome .* span)
    return lower_bounds .+ displacement
end
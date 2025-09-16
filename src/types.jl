abstract type Item{T,N,A} <: AbstractArray{T,N} end

Base.parent(x::Item) = x.data

abstract type AbstractRaster{T,N,A} <: Item{T,N,A} end

abstract type AbstractImage{T,N,A} <: AbstractRaster{T,N,A} end

abstract type AbstractMask{T,N,A} <: AbstractRaster{T,N,A} end

struct Image{T,N,A} <: AbstractImage{T,N,A}
    data::A
    Image(x::AbstractArray{T,N}) where {T,N} = new{T,N,typeof(x)}(x)
    #Image(x::AbstractArray{<:ImageCore.Colorant,N}) where N = Image(image2tensor(x))
end

struct Mask{T,N,A} <: AbstractMask{T,N,A}
    data::A
    Mask(x::AbstractArray{T,N}) where {T,N} = new{T,N,typeof(x)}(x)
end

struct NoOp{T}
    data::T
end

Base.size(x::Item, args...) = size(parent(x), args...)

function Base.getindex(x::I, i...) where {I<:Item}
    result = Base.getindex(parent(x), i...)
    return result isa AbstractArray ? I.name.wrapper(result) : result
end

function Base.view(x::I, i...) where {I<:Item}
    result = Base.view(parent(x), i...)
    return result isa AbstractArray ? I.name.wrapper(result) : result
end

Base.setindex!(x::Item, v, i...) = Base.setindex!(parent(x), v, i...)

Base.permutedims(x::I, perm) where {I<:Item} = I.name.wrapper(permutedims(parent(x), perm))

Base.IndexStyle(::Type{<:Item{T,N,A}}) where {T,N,A} = Base.IndexStyle(A)

Base.similar(x::I, ::Type{T}, dims::Dims) where {I<:Item,T} = I.name.wrapper(Base.similar(parent(x), T, dims))

Base.BroadcastStyle(::Type{<:I}) where {I<:Item} = Broadcast.ArrayStyle{I}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{I}}, ::Type{T}) where {I<:Item,T}
    return I.name.wrapper(similar(Array{T,length(axes(bc))}, axes(bc)))
end

function modify(f::Function, x::I) where {I<:Item}
    return I.name.wrapper(f(parent(x)))
end

function ImageCore.channelview(x::AbstractRaster) 
    return modify(x -> Array(ImageCore.channelview(x)), x)
end

function ImageCore.colorview(C::Type{<:ImageCore.Colorant}, x::AbstractRaster) 
    return modify(x -> Array(ImageCore.colorview(C, x)), x)
end
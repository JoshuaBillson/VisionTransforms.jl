abstract type DType{T,N} <: AbstractArray{T,N} end

abstract type AbstractImage{T,N} <: DType{T,N} end

abstract type AbstractMask{T,N} <: DType{T,N} end

struct Image2D{T} <: AbstractImage{T,3}
    data::Array{T,3}
    Image2D(x::AbstractArray{<:Any,3}) = Image2D(Array(x))
    Image2D(x::AbstractArray{<:ImageCore.Colorant,2}) = Image2D(image2tensor(x))
    Image2D(x::Array{T,3}) where T = new{T}(x)
    Image2D{T}(x::Array{T,3}) where T = new{T}(x)
end

Base.parent(x::Image2D) = x.data

channeldim(::Image2D) = 3

struct Image3D{T} <: AbstractImage{T,4}
    data::Array{T,4}
    Image3D(x::AbstractArray{<:Any,4}) = Image3D(Array(x))
    Image3D(x::Array{T,4}) where T = new{T}(x)
    Image3D{T}(x::Array{T,4}) where T = new{T}(x)
end

Base.parent(x::Image3D) = x.data

channeldim(::Image3D) = 4

struct Series2D{T} <: AbstractImage{T,4}
    data::Array{T,4}
    Series2D(x::AbstractArray{<:Any,4}) = Series2D(Array(x))
    Series2D(x::Array{T,4}) where T = new{T}(x)
    Series2D{T}(x::Array{T,4}) where T = new{T}(x)
    function Series2D(x::AbstractVector{<:AbstractArray{<:ImageCore.Colorant,2}})
        @pipe map(image2tensor, x) |> map(x -> unsqueeze(x, 4), _) |> cat(_..., dims=4)
    end
end

Base.parent(x::Series2D) = x.data

channeldim(::Series2D) = 3

struct Mask2D{T} <: AbstractMask{T,3}
    data::Array{T,3}
    Mask2D(x::AbstractArray{<:Any,3}) = Mask2D(Array(x))
    Mask2D(x::Array{T,3}) where T = new{T}(x)
    Mask2D{T}(x::Array{T,3}) where T = new{T}(x)
end

Base.parent(x::Mask2D) = x.data

channeldim(::Mask2D) = 3

struct Mask3D{T} <: AbstractMask{T,4}
    data::Array{T,4}
    Mask3D(x::AbstractArray{<:Any,4}) = Mask3D(Array(x))
    Mask3D(x::Array{T,4}) where T = new{T}(x)
    Mask3D{T}(x::Array{T,4}) where T = new{T}(x)
end

Base.parent(x::Mask3D) = x.data

channeldim(::Mask3D) = 4

struct NoOp{T}
    data::T
end

Base.parent(x::NoOp) = x.data

Base.size(x::DType) = size(parent(x))

Base.getindex(x::DType, i::Int) = parent(x)[i]

Base.setindex!(x::DType, v, i::Int) = Base.setindex!(parent(x), v, i)

Base.IndexStyle(::Type{<:D}) where {D<:DType} = IndexLinear()

Base.similar(x::D, ::Type{T}, dims::Dims) where {D<:DType,T} = D.name.wrapper(Base.similar(parent(x), T, dims))

Base.BroadcastStyle(::Type{<:D}) where {D<:DType} = Broadcast.ArrayStyle{D}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{D}}, ::Type{T}) where {D<:DType,T}
    return D.name.wrapper(similar(Array{T}, axes(bc)))
end

modify(f::Function, x::T) where {T <: DType} = T.name.wrapper(f(parent(x)))

Base.mapslices(f, a::T; dims) where {T <: DType} = modify(x -> mapslices(f, x; dims), a)

Base.selectdim(a::T, d::Integer, i) where {T <: DType} = modify(x -> selectdim(x, d, i), a)
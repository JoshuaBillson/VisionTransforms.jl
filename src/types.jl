abstract type DType{T,N} <: AbstractArray{T,N} end

abstract type AbstractImage{T,N} <: DType{T,N} end

abstract type AbstractMask{T,N} <: DType{T,N} end

struct Image2D{T} <: AbstractImage{T,4}
    data::Array{T,4}
    Image2D(x::AbstractArray{<:Any,4}) = Image2D(Array(x))
    Image2D(x::AbstractArray{<:Images.Colorant,2}) = Image2D(image2tensor(x))
    Image2D(x::Array{T,4}) where T = new{T}(x)
end

struct Image3D{T} <: AbstractImage{T,5}
    data::Array{T,5}
    Image3D(x::AbstractArray{<:Any,5}) = Image3D(Array(x))
    Image3D(x::Array{T,5}) where T = new{T}(x)
end


struct Series2D{T} <: AbstractImage{T,5}
    data::Array{T,5}
    Series2D(x::AbstractArray{<:Any,5}) = Series2D(Array(x))
    Series2D(x::Array{T,5}) where T = new{T}(x)
    function Series2D(x::AbstractVector{<:AbstractArray{<:Images.Colorant,2}})
        @pipe map(image2tensor, x) |> map(x -> unsqueeze(x, 4), _) |> cat(_..., dims=4)
    end
end

struct Mask2D{T} <: AbstractMask{T,4}
    data::Array{T,4}
    Mask2D(x::AbstractArray{<:Any,4}) = Mask2D(Array(x))
    Mask2D(x::Array{T,4}) where T = new{T}(x)
end

struct Mask3D{T} <: AbstractMask{T,5}
    data::Array{T,5}
    Mask3D(x::AbstractArray{<:Any,5}) = Mask3D(Array(x))
    Mask3D(x::Array{T,5}) where T = new{T}(x)
end

struct NoOp{T}
    data::T
end

Base.parent(x::NoOp) = x.data

for dtype = (:Image2D, :Image3D, :Series2D, :Mask2D, :Mask3D)
    @eval Base.parent(x::$dtype) = x.data

    @eval Base.size(x::$dtype) = size(x.data)

    @eval Base.getindex(x::$dtype, i::Int) = x.data[i]

    @eval Base.setindex!(x::$dtype, v, i::Int) = Base.setindex!(x.data, v, i)

    @eval Base.IndexStyle(::Type{<:$dtype}) = IndexLinear()

    @eval Base.similar(x::$dtype, ::Type{T}, dims::Dims) where {T} = $dtype(Base.similar(x.data, T, dims))

    @eval Base.BroadcastStyle(::Type{<:$dtype}) = Broadcast.ArrayStyle{$dtype}()

    @eval begin 
        function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{$dtype}}, ::Type{T}) where T
            return $dtype(similar(Array{T}, axes(bc)))
        end
    end
end
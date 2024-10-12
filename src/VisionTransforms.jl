module VisionTransforms

import Images, Distributions
using Interpolations: Constant, Linear
using Random
using Statistics
using Match
using ArgCheck: @argcheck
using Pipe: @pipe

include("utils.jl")

include("methods.jl")
export image2tensor, tensor2image, imresize, crop, flipX, flipY, rot90

include("transforms.jl")
export DType, AbstractImage, AbstractMask, Image2D, Image3D, Mask2D, Mask3D, Series2D, NoOp, AbstractTransform
export apply_transform, apply
export Resize, Scale, Normalize, PerImageNormalize, RandomCrop, FlipX, FlipY, Rot90, ComposedTransform

end

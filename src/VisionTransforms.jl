module VisionTransforms

import Images
using Interpolations: Constant, Linear
using Random
using Match
using Pipe: @pipe

include("methods.jl")
export image2tensor, tensor2image, imresize, crop, flipX, flipY, rot90

include("transforms.jl")
export Image2D, Mask2D
export Mask, Image, NoOp, AbstractTransform
export apply_transform, apply
export Resize, RandomCrop, FlipX, FlipY, Rot90, ComposedTransform

end

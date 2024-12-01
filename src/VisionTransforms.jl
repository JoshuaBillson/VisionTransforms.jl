module VisionTransforms

import Images, Rasters, Distributions
using Interpolations: Constant, Linear
using Random
using Statistics
using Match
using ArgCheck: @argcheck
using Pipe: @pipe

include("utils.jl")
include("types.jl")

include("methods.jl")
export image2tensor, tensor2image, raster2tensor, imresize, linear_stretch, per_image_linear_stretch, normalize, per_image_normalize
export crop, center_crop, random_crop, flipX, flipY, rot90, color_jitter, invert, solarize, add_noise, multiply_noise

include("transforms.jl")
export DType, AbstractImage, AbstractMask, Image2D, Image3D, Mask2D, Mask3D, Series2D, NoOp, AbstractTransform
export transform, apply
export Resize, Scale, PerImageScale, Normalize, PerImageNormalize, RandomCrop, FlipX, FlipY, Rot90, ComposedTransform
export ColorJitter

end

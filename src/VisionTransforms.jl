module VisionTransforms

import ImageCore, ImageFiltering, ImageTransformations, Distributions
using Random, Statistics, Match
using Interpolations: Constant, Linear
using ArgCheck: @argcheck
using Pipe: @pipe

include("utils.jl")
include("types.jl")
export DType, AbstractImage, AbstractMask, Image2D, Image3D, Mask2D, Mask3D, Series2D, NoOp

include("methods.jl")
export image2tensor, tensor2image, imresize, linear_stretch, per_image_linear_stretch, normalize, per_image_normalize
export crop, center_crop, random_crop, flipX, flipY, rot90, color_jitter, invert, solarize, add_noise, multiply_noise
export center_zoom, random_zoom, grayscale, adjust_brightness, adjust_contrast, adjust_hue, blur, sharpen

include("transforms.jl")
export AbstractTransform, transform, apply
export Resize, Scale, PerImageScale, Normalize, PerImageNormalize, RandomCrop, FlipX, FlipY, Rot90, ComposedTransform
export ColorJitter, TrivialAugment

end

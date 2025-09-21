module VisionTransforms

import ImageCore, ImageFiltering, ImageTransformations, Distributions
using Random, Statistics, Match
using Interpolations: Constant, Linear
using ArgCheck: @argcheck
using Pipe: @pipe

include("utils.jl")

include("types.jl")
export Item, AbstractRaster, AbstractImage, AbstractMask, Image, Mask, NoOp

include("methods/affine.jl")
export CropFrom, FromCenter, FromOrigin, FromRandom
export sample_tile, crop, zoom, imresize, flip, rot90

include("methods/color.jl")
export grayscale, adjust_brightness, adjust_contrast, color_jitter, permute_channels, invert_color, solarize

include("methods/batched.jl")
export MixUp

include("methods.jl")

include("transforms/transforms.jl")
export AbstractTransform, apply

include("transforms/affine.jl")
export Resize, Crop, CenterCrop, RandomCrop, OriginCrop, Zoom, CenterZoom, RandomZoom
export Flip, FlipX, FlipY, FlipZ, Rot90

include("transforms/color.jl")
export ColorJitter, InvertColor, Grayscale, Solarize, PermuteChannels

include("transforms/composite.jl")
export Scale, PerImageScale, Normalize, PerImageNormalize, ComposedTransform, TrivialAugment, OneOf

#include("recipe.jl")

end

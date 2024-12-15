var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = VisionTransforms","category":"page"},{"location":"#VisionTransforms","page":"Home","title":"VisionTransforms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for VisionTransforms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [VisionTransforms]","category":"page"},{"location":"#VisionTransforms.CenterCrop","page":"Home","title":"VisionTransforms.CenterCrop","text":"CenterCrop(size::Int)\nCenterCrop(size::Tuple{Int,Int})\n\nCrop a tile equal to size from the center of the input array.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.ColorJitter","page":"Home","title":"VisionTransforms.ColorJitter","text":"ColorJitter(;contrast=0.5:0.1:1.5, brightness=-0.8:0.1:0.8)\n\nApply a random color jittering transformations (contrast and brightness adjustments) according to the formula α * x + β * M, where α is contrast, β is brightness,  and M is either the mean or maximum value of x.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.ComposedTransform","page":"Home","title":"VisionTransforms.ComposedTransform","text":"ComposedTransform(transforms...)\n\nApply transforms to the input in the same order as they are given.\n\nExample\n\njulia> r = Raster(rand(256,256, 3), (X,Y,Band));\n\njulia> t = Resample(2.0) |> Tensor();\n\njulia> apply(t, Image(), r, 123) |> size\n(512, 512, 3, 1)\n\njulia> apply(t, Image(), r, 123) |> typeof\nArray{Float32, 4}\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.FlipX","page":"Home","title":"VisionTransforms.FlipX","text":"FlipX(p)\n\nApply a random horizontal flip with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.FlipY","page":"Home","title":"VisionTransforms.FlipY","text":"FlipY(p)\n\nApply a random vertical flip with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.Normalize","page":"Home","title":"VisionTransforms.Normalize","text":"Normalize(;mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n\nNormalize the channels to have a mean of 0 and a standard deviation of 1.\n\nParameters\n\nmean: The channel-wise mean of the input data (uses the ImageNet mean by default).\nstd: The channel-wise standard deviation of the input data (uses the ImageNet std by default).\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.PerImageNormalize","page":"Home","title":"VisionTransforms.PerImageNormalize","text":"PerImageNormalize()\n\nNormalize the channels to have a mean of 0 and a standard deviation of 1 based on statistics calculated for each image in a batch.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.PerImageScale","page":"Home","title":"VisionTransforms.PerImageScale","text":"PerImageScale(;lower=0.02, upper=0.98)\n\nApply a linear stretch to scale all values to the range [0, 1]. The arguments lower and upper specify the percentiles at which to define the lower and upper bounds from each channel in the source image. Values that either fall below lower or above upper will be clamped.\n\nParameters\n\nlower: The quantile to use as the lower-bound in the source array.\nupper: The quantile to use as the upper-bound in the source array.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.RandomCrop","page":"Home","title":"VisionTransforms.RandomCrop","text":"RandomCrop(size::Int)\nRandomCrop(size::Tuple{Int,Int})\n\nCrop a randomly placed tile equal to size from the input array.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.RandomInvert","page":"Home","title":"VisionTransforms.RandomInvert","text":"RandomInvert(p)\n\nApply a random color inversion with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.RandomSolarize","page":"Home","title":"VisionTransforms.RandomSolarize","text":"RandomSolarize(p; threshold=0.75)\n\nRandomly solarize an image with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.Resize","page":"Home","title":"VisionTransforms.Resize","text":"Resize(sz::Tuple)\n\nResample x according to the specified scale. Mask types will always be resampled with :near interpolation, whereas Images will be resampled with  either :bilinear (scale > 1) or :average (scale < 1).\n\nParameters\n\nx: The image/mask to be resampled.\nsz: The size of the output image.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.Rot90","page":"Home","title":"VisionTransforms.Rot90","text":"Rot90(p)\n\nApply a random 90 degree rotation with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.Scale","page":"Home","title":"VisionTransforms.Scale","text":"Scale(lower::Vector{<:Real}, upper::Vector{<:Real})\n\nApply a linear stretch to scale all values to the range [0, 1]. The arguments lower and upper specify the lower and upper bounds from each channel in the source image. Values that  either fall below lower or above upper will be clamped.\n\nParameters\n\nlower: The lower-bounds to use for each channel in the source image.\nupper: The upper-bounds to use for each channel in the source image.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.add_noise-Tuple","page":"Home","title":"VisionTransforms.add_noise","text":"add_noise([rng], dist::Distributions.Distribution, x::AbstractArray; correlated=true)\n\nAdd noise generated by the distribution dist to the image tensor x.\n\nParameters\n\nrng: A random number generator to make the outcome reproducible.\ndist: A Distributions.Distribution object from which to sample the noise.\nx: A tensor containing a 2D or 3D image or image series.\ncorrelated: If true, applies the same noise value to each channel in the image.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.adjust_brightness-Union{Tuple{T}, Tuple{AbstractArray{T}, Real}} where T<:Real","page":"Home","title":"VisionTransforms.adjust_brightness","text":"adjust_brightness(x::AbstractArray, brightness::Real)\n\nAdjust the brightness of x by brightness.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.adjust_contrast-Tuple{AbstractImage, Real}","page":"Home","title":"VisionTransforms.adjust_contrast","text":"adjust_contrast(x::AbstractImage, contrast::Real)\nadjust_contrast(x::AbstractArray, contrast::Real, channeldim::Int)\n\nAdjust the contrast of x by contrast.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.adjust_hue-Tuple{AbstractImage, Real}","page":"Home","title":"VisionTransforms.adjust_hue","text":"adjust_hue(x::AbstractImage, strength::Real)\nadjust_hue(x::AbstractArray, strength::Real, channeldim::Int)\n\nAdjust the hue of x by jittering the relative brightness of each channel according to strength.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.blur-Tuple{Image2D, Real}","page":"Home","title":"VisionTransforms.blur","text":"blur(x::AbstractImage, strength::Real)\nblur(x::AbstractArray, strength::Real, channeldim::Int)\n\nBlur the image x by applying a gaussian filter with a standard deviation of strength.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.center_crop-Tuple{AbstractArray, Int64}","page":"Home","title":"VisionTransforms.center_crop","text":"center_crop(x::AbstractArray, sz::Int)\ncenter_crop(x::AbstractArray, sz::Tuple{Int,Int})\n\nCrop x to the size specified by sz from the center.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.center_zoom-Tuple{DType, Int64}","page":"Home","title":"VisionTransforms.center_zoom","text":"center_zoom(x::DType, zoom_strength::Int)\ncenter_zoom(x::AbstractArray, zoom_strength::Int, method::Symbol, channeldim::Int)\n\nZoom to the center of x by a factor of zoom_strength.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.color_jitter-Tuple{Int64, Image2D, Any, Any}","page":"Home","title":"VisionTransforms.color_jitter","text":"color_jitter(seed::Int, x::Image2D, contrast, brightness; kw...)\ncolor_jitter(seed::Int, x::Image3D, contrast, brightness; kw...)\ncolor_jitter(seed::Int, x::Series2D, contrast, brightness; kw...)\ncolor_jitter(seed::Int, x::AbstractArray, contrast, brightness, dims; usemax=true)\n\nApplies random color jittering transformations (contrast and brightness adjustments) to  input images or data series according to the formula α * x + β * M, where α is contrast, β is brightness, and M is either the mean or maximum value of x.\n\nParameters\n\nrng: A random number generator to make the outcome reproducible.\ndist: A Distributions.Distribution object from which to sample the noise.\nx: A tensor containing a 2D or 3D image or image series.\ncorrelated: If true, applies the same noise value to each channel in the image.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.crop","page":"Home","title":"VisionTransforms.crop","text":"crop(x, sz::Int, ul=(1,1))\ncrop(x, sz::Tuple{Int,Int}, ul=(1,1))\n\nCrop a tile equal to size out of x with an upper-left corner defined by ul.\n\n\n\n\n\n","category":"function"},{"location":"#VisionTransforms.flipX-Tuple{AbstractMatrix}","page":"Home","title":"VisionTransforms.flipX","text":"flipX(x)\n\nFlip the image x across the horizontal axis.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.flipY-Tuple{AbstractMatrix}","page":"Home","title":"VisionTransforms.flipY","text":"flipY(x)\n\nFlip the image x across the vertical axis.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.grayscale-Tuple{AbstractImage}","page":"Home","title":"VisionTransforms.grayscale","text":"grayscale(x::AbstractImage)\ngrayscale(x::AbstractArray, channeldim::Int)\n\nConvert x to a grayscale image.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.imresize-Tuple{Mask2D, Tuple}","page":"Home","title":"VisionTransforms.imresize","text":"imresize(img::AbstractMask, sz::Tuple)\nimresize(img::AbstractImage, sz::Tuple)\nimresize(img::AbstractArray, sz::Tuple, method::Symbol, channeldim::Int)\n\nResize img to sz with the specified resampling method.\n\nParameters\n\nimg: The image to be resized.\nsz: The width and height of the output as a tuple.\nmethod: Either :nearest or :bilinear.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.invert-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T<:Real","page":"Home","title":"VisionTransforms.invert","text":"invert(x::AbstractArray{<:Real})\n\nInvert the values of x according to the formula maximum(x) .- x.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.linear_stretch-Tuple{Image2D, Vector{<:Real}, Vector{<:Real}}","page":"Home","title":"VisionTransforms.linear_stretch","text":"linear_stretch(x::Image2D, lower::Vector{<:Real}, upper::Vector{<:Real})\nlinear_stretch(x::Image3D, lower::Vector{<:Real}, upper::Vector{<:Real})\nlinear_stretch(x::Series2D, lower::Vector{<:Real}, upper::Vector{<:Real})\nlinear_stretch(x::AbstractArray, lower::Vector{<:Real}, upper::Vector{<:Real}, channel_dim::Int)\n\nPerform a linear histogram stretch on x such that lower is mapped to 0 and upper is mapped to 1. Values outside the interval [lower, upper] will be clamped.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.multiply_noise-Tuple","page":"Home","title":"VisionTransforms.multiply_noise","text":"multiply_noise([rng], dist::Distributions.Distribution, x::AbstractArray; correlated=true)\n\nMultiply noise generated by the distribution dist to the image tensor x.\n\nParameters\n\nrng: A random number generator to make the outcome reproducible.\ndist: A Distributions.Distribution object from which to sample the noise.\nx: A tensor containing a 2D or 3D image or image series.\ncorrelated: If true, applies the same noise value to each channel in the image.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.normalize-Tuple{AbstractArray{<:Integer}, Vararg{Any}}","page":"Home","title":"VisionTransforms.normalize","text":"normalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=1)\n\nNormalize the input array with respect to the specified dimension so that the mean is 0 and the standard deviation is 1.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to normalize the input array.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.per_image_linear_stretch-Tuple{Image2D, Real, Real}","page":"Home","title":"VisionTransforms.per_image_linear_stretch","text":"per_image_linear_stretch(x::Image2D, lower::Real, upper::Real)\nper_image_linear_stretch(x::Image3D, lower::Real, upper::Real)\nper_image_linear_stretch(x::Series2D, lower::Real, upper::Real)\nper_image_linear_stretch(x::AbstractArray, lower::Real, upper::Real, channel_dim::Int)\n\nApply a linear stretch to scale all values to the range [0, 1]. The arguments lower and upper specify the percentiles at which to define the lower and upper bounds from each channel in the source image. Values that either fall below lower or above upper will be clamped.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.per_image_normalize-Tuple{Image2D}","page":"Home","title":"VisionTransforms.per_image_normalize","text":"per_image_normalize(x::Image2D)\nper_image_normalize(x::Image3D)\nper_image_normalize(x::Series2D)\nper_image_normalize(x::AbstractArray, dims::Tuple)\n\nNormalize the input array so that the mean and standard deviation of each channel is 0 and 1, respectively. Unlike normalize, this method will compute new statistics for each image in x. This is more computationally expensive, but may be more suitable when there is significant domain shift between train and test images. \n\nParameters\n\nx: A tensor containing one or more 2D or 3D images or 2D image series.\ndims: The dimensions over which to compute image statistics.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.random_crop-Tuple{Int64, AbstractArray, Tuple{Int64, Int64}}","page":"Home","title":"VisionTransforms.random_crop","text":"random_crop(seed::Int, x::AbstractArray, sz::Tuple{Int,Int})\n\nCrop a randomly placed tile equal to sz from the array x.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.random_zoom-Tuple{Int64, AbstractArray, AbstractVector{<:Real}, Vararg{Any}}","page":"Home","title":"VisionTransforms.random_zoom","text":"random_zoom(seed::Int, x::DType, zoom_strength::Real)\nrandom_zoom(seed::Int, x::AbstractArray, zoom_strength::AbstractVector{<:Real}, args...)\nrandom_zoom(seed::Int, x::AbstractArray, zoom_strength::Real, method::Symbol, channeldim::Int)\n\nZoom to a random location in x by a factor of zoom_strength.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.rot90-Tuple{AbstractArray{<:Any, 4}}","page":"Home","title":"VisionTransforms.rot90","text":"rot90(x)\n\nRotate the image x by 90 degress. \n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.sharpen-Tuple{Image2D, Real}","page":"Home","title":"VisionTransforms.sharpen","text":"sharpen(x::AbstractImage, strength::Real)\nsharpen(x::AbstractArray, strength::Real, channeldim::Int)\n\nSharpen the image x by applying a high-frequency-boosting filter.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.solarize-Tuple{AbstractArray{<:Real}}","page":"Home","title":"VisionTransforms.solarize","text":"solarize(x::AbstractArray{<:Real}; threshold=0.75)\n\nSolarize x by inverting all values above threshold.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.transform-Tuple{AbstractTransform, Type{<:NoOp}, Any}","page":"Home","title":"VisionTransforms.transform","text":"transform(t::AbstractTransform, dtype, x)\ntransform(t::AbstractTransform, dtypes::Tuple, x::Tuple)\n\nApply the transformation t to the input x with data type dtype.\n\n\n\n\n\n","category":"method"}]
}

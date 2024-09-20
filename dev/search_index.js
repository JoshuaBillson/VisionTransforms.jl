var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = VisionTransforms","category":"page"},{"location":"#VisionTransforms","page":"Home","title":"VisionTransforms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for VisionTransforms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [VisionTransforms]","category":"page"},{"location":"#VisionTransforms.ComposedTransform","page":"Home","title":"VisionTransforms.ComposedTransform","text":"ComposedTransform(transforms...)\n\nApply transforms to the input in the same order as they are given.\n\nExample\n\njulia> r = Raster(rand(256,256, 3), (X,Y,Band));\n\njulia> t = Resample(2.0) |> Tensor();\n\njulia> apply(t, Image(), r, 123) |> size\n(512, 512, 3, 1)\n\njulia> apply(t, Image(), r, 123) |> typeof\nArray{Float32, 4}\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.FlipX","page":"Home","title":"VisionTransforms.FlipX","text":"FlipX(p)\n\nApply a random horizontal flip with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.FlipY","page":"Home","title":"VisionTransforms.FlipY","text":"FlipY(p)\n\nApply a random vertical flip with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.RandomCrop","page":"Home","title":"VisionTransforms.RandomCrop","text":"RandomCrop(size::Int)\nRandomCrop(size::Tuple{Int,Int})\n\nCrop a randomly placed tile equal to size from the input array.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.Resize","page":"Home","title":"VisionTransforms.Resize","text":"Resize(sz::Tuple)\n\nResample x according to the specified scale. Mask types will always be resampled with :near interpolation, whereas Images will be resampled with  either :bilinear (scale > 1) or :average (scale < 1).\n\nParameters\n\nx: The raster/stack to be resampled.\nscale: The size of the output with respect to the input.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.Rot90","page":"Home","title":"VisionTransforms.Rot90","text":"Rot90(p)\n\nApply a random 90 degree rotation with probability p.\n\n\n\n\n\n","category":"type"},{"location":"#VisionTransforms.apply_transform-Tuple{AbstractTransform, Any}","page":"Home","title":"VisionTransforms.apply_transform","text":"transform(t::AbstractTransform, dtype::DType, x)\ntransform(t::AbstractTransform, dtypes::Tuple, x::Tuple)\n\nApply the transformation t to the input x with data type dtype.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.crop","page":"Home","title":"VisionTransforms.crop","text":"crop(x, sz::Int, ul=(1,1))\ncrop(x, sz::Tuple{Int,Int}, ul=(1,1))\n\nCrop a tile equal to size out of x with an upper-left corner defined by ul.\n\n\n\n\n\n","category":"function"},{"location":"#VisionTransforms.flipX-Tuple{AbstractMatrix}","page":"Home","title":"VisionTransforms.flipX","text":"flipX(x)\n\nFlip the image x across the horizontal axis.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.flipY-Tuple{AbstractMatrix}","page":"Home","title":"VisionTransforms.flipY","text":"flipY(x)\n\nFlip the image x across the vertical axis.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.imresize-Union{Tuple{N}, Tuple{AbstractArray{<:Real, N}, Tuple}} where N","page":"Home","title":"VisionTransforms.imresize","text":"resize(img::AbstractArray, sz::Tuple{Int,Int}; method=:bilinear)\n\nResize img to sz with the specified resampling method.\n\nParameters\n\nimg: The image to be resized.\nsz: The width and height of the output as a tuple.\nmethod: Either :nearest or :bilinear.\n\n\n\n\n\n","category":"method"},{"location":"#VisionTransforms.rot90-Tuple{AbstractArray{<:Any, 4}}","page":"Home","title":"VisionTransforms.rot90","text":"rot90(x)\n\nRotate the image x by 90 degress. \n\n\n\n\n\n","category":"method"}]
}
using VisionTransforms
using Test

@testset "VisionTransforms.jl" begin
    img = rand(Float32, 256, 256, 3) |> Image2D
    mask = rand([0, 1], 256, 256, 1) |> Mask2D

    # Image Resize
    @test size(imresize(img, (112,112))) == (112,112,3)
    @test size(imresize(mask, (112,112))) == (112,112,1)
    @test all(x -> x in (0, 1), imresize(mask, (112,112)))
    @test size(imresize(img, (512,512))) == (512,512,3)
    @test size(imresize(mask, (512,512))) == (512,512,1)
    @test all(x -> x in (0, 1), imresize(mask, (512,512)))
    @test imresize(img, (256,256)) == img
    @test imresize(mask, (256,256)) == mask

    # Image Crop
    @test size(crop(img, 128, (1,1))) == (128,128,3)
    @test size(crop(mask, 128, (1,1))) == (128,128,1)
    @test size(crop(img, 128, (129,129))) == (128,128,3)
    @test_throws ArgumentError crop(img, 257, (1,1))
    @test_throws ArgumentError crop(img, 0, (1,1))
    @test_throws ArgumentError crop(img, 128, (-1,1))
    @test_throws ArgumentError crop(img, 128, (1,-1))
    @test_throws ArgumentError crop(img, 128, (129,130))
    @test_throws ArgumentError crop(img, 128, (0,0))
end

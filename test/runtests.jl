using VisionTransforms
using Test

@testset "VisionTransforms.jl" begin
    img = rand(Float32, 256, 256, 3)
    mask = rand([0, 1], 256, 256, 1)

    # Image Resize
    t1 = Resize((112,112))
    t2 = Resize((512,512))
    t3 = Resize((256,256))
    @test size(transform(t1, Image2D, img)) == (112,112,3)
    @test size(transform(t1, Mask2D, mask)) == (112,112,1)
    @test all(x -> x in (0, 1), transform(t1, Mask2D, mask))
    @test size(transform(t2, Image2D, img)) == (512,512,3)
    @test size(transform(t2, Mask2D, mask)) == (512,512,1)
    @test all(x -> x in (0, 1), transform(t2, Mask2D, mask))
    @test transform(t3, Image2D, img) == img
    @test transform(t3, Mask2D, mask) == mask

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

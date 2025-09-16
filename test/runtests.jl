using VisionTransforms
using Test

@testset "VisionTransforms.jl" begin
    img = rand(Float32, 256, 256, 3)
    mask = rand([0, 1], 256, 256, 1)

    # Image Resize
    t1 = Resize((112,112))
    t2 = Resize((512,512))
    t3 = Resize((256,256))
    @test size(transform(t1, Image, img)) == (112,112,3)
    @test size(transform(t1, Mask, mask)) == (112,112,1)
    @test all(x -> x in (0, 1), transform(t1, Mask, mask))
    @test size(transform(t2, Image, img)) == (512,512,3)
    @test size(transform(t2, Mask, mask)) == (512,512,1)
    @test all(x -> x in (0, 1), transform(t2, Mask, mask))
    @test transform(t3, Image, img) == img
    @test transform(t3, Mask, mask) == mask

    # Image Crop
    for from in [FromCenter(), FromRandom(), FromOrigin()]
        @test size(crop(from, (128,128), img, 123)) == (128,128,3)
        @test size(crop(from, (128,128), mask, 123)) == (128,128,1)
    end

    # Trivial Augment
    #for i in 1:20
    #    @test size(transform(TrivialAugment(), Image, img)) == (256,256,3)
    #    @test size(transform(TrivialAugment(), Mask, mask)) == (256,256,1)
    #    @test all(x -> x in (0, 1), transform(TrivialAugment(), Mask, mask))
    #end
end

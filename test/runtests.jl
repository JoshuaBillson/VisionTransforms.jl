using VisionTransforms
using Test

@testset "VisionTransforms.jl" begin
    img = rand(Float32, 256, 256, 3)
    mask = rand([0, 1], 256, 256, 1)

    # Image Resize
    t1 = Resize((112,112))
    t2 = Resize((512,512))
    t3 = Resize((256,256))
    @test size(t1(Image => img)) == (112,112,3)
    @test size(t1(Mask => mask)) == (112,112,1)
    @test all(x -> x in (0, 1), t1(Mask => mask))
    @test size(t2(Image => img)) == (512,512,3)
    @test size(t2(Mask => mask)) == (512,512,1)
    @test all(x -> x in (0, 1), t2(Mask => mask))
    @test t3(Image => img) == img
    @test t3(Mask => mask) == mask

    # Image Crop
    for t in [CenterCrop((128,128)), RandomCrop((128,128)), OriginCrop((128,128))]
        @test size(t(Image => img)) == (128,128,3)
        @test size(t(Mask => mask)) == (128,128,1)
    end

    # Zoom
    for t in [CenterZoom(strength=[2]), RandomZoom(strength=[2])]
        @test size(t(Image => img)) == (256,256,3)
        @test size(t(Mask => mask)) == (256,256,1)
        @test t(Image => img) != img
        @test t(Mask => mask) != mask
    end

    # Test Consistency
    t1 = RandomCrop((128,128)) |> FlipX(;p=0.5) |> FlipY(;p=0.5) |> Rot90(;p=0.25) |> ColorJitter()
    t2 = RandomCrop((128,128)) |> FlipX(;p=0.5) |> FlipY(;p=0.5) |> Rot90(;p=0.25)
    for i in 1:5
        x1, x2, x3 = t1((Image,Image,Mask) .=> (img,img,img))
        @test x1 == x2
        @test x1 != x3
        x1, x2, x3 = t2((Image,Image,Mask) .=> (img,img,img))
        @test x1 == x2
        @test x1 == x3
    end

    # Trivial Augment
    #for i in 1:20
    #    @test size(transform(TrivialAugment(), Image, img)) == (256,256,3)
    #    @test size(transform(TrivialAugment(), Mask, mask)) == (256,256,1)
    #    @test all(x -> x in (0, 1), transform(TrivialAugment(), Mask, mask))
    #end
end

using VisionTransforms
using Documenter

DocMeta.setdocmeta!(VisionTransforms, :DocTestSetup, :(using VisionTransforms); recursive=true)

makedocs(;
    modules=[VisionTransforms],
    authors="Joshua Billson",
    sitename="VisionTransforms.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/VisionTransforms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/VisionTransforms.jl",
    devbranch="main",
)

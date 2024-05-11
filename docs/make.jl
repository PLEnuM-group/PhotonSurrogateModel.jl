using PhotonSurrogateModel
using Documenter

DocMeta.setdocmeta!(PhotonSurrogateModel, :DocTestSetup, :(using PhotonSurrogateModel); recursive=true)

makedocs(;
    modules=[PhotonSurrogateModel],
    authors="Christian Haack <chr.hck@gmail.com>",
    sitename="PhotonSurrogateModel.jl",
    format=Documenter.HTML(;
        canonical="https://chrhck.github.io/PhotonSurrogateModel.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chrhck/PhotonSurrogateModel.jl",
    devbranch="main",
)

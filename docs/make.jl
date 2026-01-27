using EoVSolvers
using Documenter

DocMeta.setdocmeta!(EoVSolvers, :DocTestSetup, :(using EoVSolvers); recursive=true)

makedocs(;
    modules=[EoVSolvers],
    authors="Liam A. A. Blake <liam.blake@adelaide.edu.au>",
    sitename="EoVSolvers.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

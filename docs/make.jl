using Documenter, Gen

makedocs(
    format = :html,
    sitename = "GenTF",
    modules = [GenTF],
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/probcomp/GenTF.git",
    target = "build"
)

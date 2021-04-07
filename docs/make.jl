using Documenter
using GenTF

makedocs(
    sitename = "GenTF",
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/probcomp/GenTF.git",
    devbranch = "master"
)

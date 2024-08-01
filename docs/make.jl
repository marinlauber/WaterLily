using Documenter, WaterLily

pages = [
  "Home" => "index.md",
  "Getting Started" => "getting-started.md",
  "Function Reference" => "methods.md",
  "WaterLily" => "helper.md",
#   "WaterLily" => "WaterLily.md",
  "Developer notes" => "dev-notes.md",
]

## find all image files in examples/ dir and copy in docs/src/examples/

recursive_find(directory, pattern) =
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end

image_files = []

for pattern in [r"\.gif", r"\.jpg", r"\.png"]
    global image_files = vcat(image_files, recursive_find(joinpath(@__DIR__, "../examples"), pattern))
end

for file in image_files
    cp(file, joinpath(@__DIR__, "src/examples/"*basename.(file)), force=true)
end


## now ready to make the docs

makedocs(
    modules = [WaterLily],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://WaterLily-jl.github.io/WaterLily.jl/",
        assets=String[],
        mathengine = MathJax3()
    ),
    authors = "Gabriel Weymouth",
    sitename = "WaterLily.jl",
    pages = pages
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/WaterLily-jl/WaterLily.jl.git",
    target = "build",
    branch = "gh-pages",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#" ],
)

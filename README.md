# GenTF

## Installing for development

First, clone the repository somewhere on your filesystem (e.g. `/home/marcoct/dev/GenTF/`), and instruct the Julia package manager to add it as a dependency in the current project, in development mode:
```
julia -e 'Pkg.clone("git@github.com:probcomp/GenTF.git")'
julia -e 'julia -e 'Pkg.develop(Pkg.PackageSpec(path="/home/marcoct/dev/GenTF"))'
```

Then, build the package and its dependencies, using a standalone Python instead of the system Python:
```
julia -e 'ENV["PYTHON"] = ""; Pkg.build("GenTF")'
```

To build with GPU support (requires CUDA, see TensorFlow.jl for more details), use:
```
julia -e 'ENV["PYTHON"] = ""; ENV["TF_USE_GPU"] = 1; Pkg.build("GenTF")'
```

Test it:
```
julia -e 'Pkg.test("GenTF")'
```

See the docker file for an example build that does not use GPU.
Also see TensorFlow.jl for issues building TensorFlow.

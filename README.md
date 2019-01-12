# GenTF [![Build Status](https://travis-ci.org/probcomp/GenTF.svg?branch=master)](https://travis-ci.org/probcomp/GenTF)

TensorFlow plugin for [Gen](https://github.com/probcomp/Gen).

## Installation

From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
add TensorFlow#8a28acb
add https://github.com/probcomp/GenTF
```
In a Julia REPL, build TensorFlow.jl to use an appropriate Python version. We have succesfully tested the following:
```
ENV["PYTHON"] = ""; ENV["CONDA_JL_VERSION"] = "2"; using Pkg; Pkg.build("TensorFlow")
```

To enable TensorFlow to use the GPU, use:
```
ENV["PYTHON"] = ""; ENV["CONDA_JL_VERSION"] = "2"; ENV["TF_USE_GPU"] = "1"; using Pkg; Pkg.build("TensorFlow")
```

See [TensorFlow.jl](https://github.com/malmaud/TensorFlow.jl) for issues building TensorFlow.

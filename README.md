# GenTF [![Build Status](https://travis-ci.org/probcomp/GenTF.svg?branch=master)](https://travis-ci.org/probcomp/GenTF)

TensorFlow plugin for [Gen](https://github.com/probcomp/Gen).

*Warning: This is rapidly evolving and currently unsupported research software. We are currently working on stabilizing the DSLs and APIs, and developing documentation and tutorials.*

## Documentation

Documentation of the development version: [https://probcomp.github.io/GenTF/dev/](https://probcomp.github.io/GenTF/dev/)

## Installation

The installation requires an installation of Python and an installation of the [`tensorflow`]((https://www.tensorflow.org/install/pip)) Python package.
We recommend creating a Python virtual environment and installing TensorFlow via pip in that environment.
In what follows, let `<python>` stand for the absolute path of a Python executable that has access to the `tensorflow` package.

From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
pkg> add https://github.com/probcomp/GenTF
```
In a Julia REPL, build the `PyCall` module so that it will use the correct Python environment:
```julia
using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyCall")
```
Also see https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version.

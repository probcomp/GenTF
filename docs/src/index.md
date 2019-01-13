# GenTF

*TensorFlow plugin for the Gen probabilistic programming system*

## Installation

The installation requires an installation of Python and an installation of the [tensorflow](https://www.tensorflow.org/install/pip) Python package.
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


## Calling the TensorFlow Python API

GenTF uses the Julia package [PyCall](https://github.com/JuliaPy/PyCall.jl) to run Python code that constructs [TensorFlow](https://www.tensorflow.org/) computation graphs using the [TensorFlow Python API](https://www.tensorflow.org/api_docs/python/).

First, import PyCall:
```julia
using PyCall
```
Then import the `tensorflow` module from the TensorFlow Python package:
```julia
@pyimport tensorflow as tf
```
To import a module from a subpackage:
```julia
@pyimport tensorflow.train as train
```
Then, you can call the TensorFlow Python API using syntax that is very close and in many cases identical to Python syntax:
```julia
W = tf.get_variable("W", dtype=tf.float32, initializer=init_W)
x = tf.placeholder(tf.float32, shape=(3,), name="x")
y = tf.squeeze(tf.matmul(W, tf.expand_dims(x, axis=1)), axis=1)
sess = tf.Session()
sess[:run](tf.global_variables_initializer())
y_val = sess[:run](y, feed_dict=Dict(x => [1., 2., 3.]))
```
Here are syntax changes that are required for common situations:

- Attributes of Python objects (including methods) are accessed using `o[:attr]` instead of `o.attr`. Therefore, to run something in a TensorFlow session `sess`, use `sess[:run](..)` instead of `sess.run(..)`.

- Where Python dictionaries would be used, use Julia dictionaries instead.

- `[1, 1, 1, 1]` constructs a Julia `Array`, which by default gets converted to a numpy array, not a Julia list. When a TensorFlow API function requires that an argument is a Python list or a tuple (e.g. the `strides` argument of [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)), use a Julia tuple: `(1, 1, 1, 1)`.

See the [PyCall README](https://github.com/JuliaPy/PyCall.jl) for the complete description of syntax differences introduced when using Julia and PyCall instead of Python.

## TensorFlow Generative Functions

A TensorFlow computation graph contains both the model(s) being trained as well as the operations that do the training.
In contrast, Gen uses a more rigid separation between models (both generative models and inference models) and the operations that act on models.
Specifically, models in Gen are defined as (functional and stateless) *generative functions*, and the operations that run the models or train the models are defined in separate Julia code.
The GenTF package allows users to construct deterministic generative functions from a TensorFlow computation graph in which each TensorFlow element is one of the following:

- A `tf.placeholder`. These play the role of *arguments* to the generative function.

- A `tf.Variable`. These play the role of the *trainable parameters* of the generative function. We will discuss how to train the parameters in section [Implementing praameter updates](#implementing-parameter-updates).

- A Tensor produced from a non-mutating operation applied to another Tensor, placeholder, or Variable (e.g. `tf.conv2d` is allowed but `tf.assign` is not). These comprise the actual computation performed by the generative function.

Note that we do not currently permit TensorFlow generative functions to use randomness.

- what happens when the model is defined
    > the global variable initializer is run..
    > TODO: should each model have its own session?
    > it's either that or we force users to manually run the 
      initialation steps (global variables initializer), then reset gradients.

    > how the gradients are accumulated

- what happens during backprop_trace
    > gradients with respect to inputs
    > gradents with respect to parameters are NOT taken

- what happens during backprop_params
    > gradients with respect to inputs
    > accumulation of gradients

## Implementing parameter updates

- how to use it with Gen:
    > show an example SGD algorithm
    > explain initialize, backprop_params, the move

## API

```@docs
TFFunction
get_param_grad_tf_var
reset_param_grads_tf_op
```

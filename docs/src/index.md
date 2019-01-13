# GenTF

*TensorFlow plugin for the Gen probabilistic programming system*

The Julia package [GenTF](https://github.com/probcomp/GenTF) allows for [Gen](https://github.com/probcomp/Gen) generative functions to invoke TensorFlow computations executed on the GPU by the TensorFlow runtime.
Users construct a TensorFlow computation using the familiar TensorFlow Python API, and then package the TensorFlow computation in a `TFFunction`, which is a type of generative function provided by GenTF.
Generative functions written in Gen's built-in modeling language can seamlessly call `TFFunction`s.
GenTF integrates Gen's automatic differentiation with TensorFlow's gradients, allowing automatic differentiation of computations that combine Julia and TensorFlow code.

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
Check that intended python environment is indeed being used with:
```julia
using PyCall; println(PyCall.python)
```
If you encounter problems, see https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version


## Calling the TensorFlow Python API

GenTF uses the Julia package [PyCall](https://github.com/JuliaPy/PyCall.jl) to invoke the [TensorFlow Python API](https://www.tensorflow.org/api_docs/python/).

First, import PyCall:
```julia
using PyCall
```
Then import the `tensorflow` Python module:
```julia
@pyimport tensorflow as tf
```
To import a module from a subpackage:
```julia
@pyimport tensorflow.train as train
```
Then, call the TensorFlow Python API with syntax that is very close to Python syntax:
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
Specifically, models in Gen are defined as (pure functional) *generative functions*, and the operations that run the models or train the models are defined in separate Julia code.
The GenTF package allows users to construct deterministic generative functions of type `TFFunction <: GenerativeFunction` from a TensorFlow computation graph in which each TensorFlow element is one of the following roles:

Role in `TFFunction`         | TensorFlow object type
:--------------------------- | :--------------------------
Argument                     | tf.Tensor produced by tf.placeholder
Trainable Parameter          | tf.Variable
Operation in Body            | tf.Tensor produced by non-mutating TensorFlow operation (e.g. tf.conv2d)
N/A                          | tf.Tensor produced by mutating TensorFlow operation (e.g. tf.assign)

TensorFlow placeholders play the role of **arguments** to the generative function.
TensorFlow Variables play the role of the **trainable parameters** of the generative function.
Their value is shared across all invocations of the generative function and is managed by the TensorFlow runtime, not Julia.
We will discuss how to train these parameters in section [Implementing parameter updates](@ref).
Tensors produced from non-mutating operations comprise the **body** of the generative function.
One of these elements (either an argument parameter, or element of the body) is designated the **return value** of the generative function.
Note that we do not currently permit TensorFlow generative functions to use randomness.

To construct a TensorFlow generative function, we first construct the TensorFlow computation graph using the TensorFlow Python API:
```julia
using Gen
using GenTF
using PyCall

@pyimport tensorflow as tf
@pyimport tensorflow.nn as nn

xs = tf.placeholder(tf.float64) # N x 784
W = tf.Variable(zeros(Float64, 784, 10))
b = tf.Variable(zeros(Float64, 10))
probs = nn.softmax(tf.add(tf.matmul(xs, W), b), axis=1) # N x 10
```

Then we construct a `TFFunction` from the TensorFlow graph objects.
The first argument to `TFFunction` is the TensorFlow session, followed by a `Vector` of trainable parameters (`W` and `b`), a `Vector` of arguments (`xs`), and finally the **return value** (`probs`).
```
sess = tf.Session()
tf_func = TFFunction(sess, [W, b], [xs], probs)
```
The return value must be a differentiable function of each argument and each parameter.
Note that the return value does *not* need to be a scalar.
TensorFlow computations for gradients with respect to the arguments and with respect to the parameters are automatically generated when constructing the `TFFunction`.

Values for the parameters are managed by the TensorFlow runtime.
The value of a trainable parameter can be obtained in Julia using the reference to the Python Variable object:
```julia
W_value = sess[:run](W)
```

### What happens during `Gen.initialize`

Suppose we run `initialize` on the `TFFunction`:
```julia
(trace, weight) = initialize(tf_func, (xs_val,), EmptyAssignment())
```

- The TensorFlow runtime computes the return value for the given values of the arguments and the current values of of the trainable parameters.

- The return value is obtained by Julia from TensorFlow and stored in the trace (it is accessible with `get_retval(trace)`).

- The given argument values are also stored in the trace (accessible with `get_args(trace)`).

Note that we pass an empty assignment to `initialize` because a `TFFunction` cannot make any random choices that could be constrained.


### What happens during `Gen.backprop_trace`

When running `backprop_trace` with a trace produced from a `TFFunction`, we must pass a gradient value for the return value.
This value should be a Julia `Array` with the same shape as the return value.
```julia
((xs_grad,), _, _) = backprop_trace(trace, EmptyAddressSet(), retval_grad)
```

- The gradients with respect to each argument are computed by the TensorFlow runtime.

- The values of the gradient are converted to Julia values and returned.

Note that we pass an empty selection because a `TFFunction` does not make any random choices that could be selected.


### What happens during `Gen.backprop_params`

When running `backprop_params` with a trace produced from a `TFFunction`, we must pass a gradient value for the return value.
This value should be a Julia `Array` with the same shape as the return value.
```julia
(xs_grad,) = backprop_params(trace, retval_grad)
```

- Like `backprop_trace`, the method returns the value of the gradient with respect to the arguments

- The gradient with respect to each trainable parameters is computed by the TensorFlow runtime.

- A **gradient accumulator** TensorFlow Variable for each trainable parameter is incremented by the corresponding gradient value.

The gradient accumulator for a parameter accumulates gradient contributions over multiple invocations of `backprop_params`.
A gradient accumulator TensorFlow Variable value can be obtained from the `TFFunction` with `get_param_grad_tf_var` (see [API](@ref) below).
The value of all gradient accumulators for a given `TFFunction` can be reset to zeros with `reset_param_grads_tf_op` (see [API](@ref) below).

## Implementing parameter updates

Updates to the trainable parameters of a `TFFunction` are also defined using the TensorFlow Python API.
For example, below we define a TensorFlow operation to apply one step of stochastic gradient descent, based on the current values of the gradient accumulators for all parameters:
```julia
opt = train.GradientDescentOptimizer(.00001)
grads_and_vars = []
push!(grads_and_vars, (tf.negative(get_param_grad_tf_var(tf_func, W)), W))
push!(grads_and_vars, (tf.negative(get_param_grad_tf_var(tf_func, b)), b))
update = opt[:apply_gradients](grads_and_vars)
```
We can then apply this update with:
```julia
sess[:run](update)
```
We can reset the gradient accumulators to zero when desired with:
```julia
sess[:run](reset_param_grads_tf_op(tf_func))
```

## Examples

See the `examples/` directory for examples that show `TFFunction`s being combined with regular Gen functions.

## API

```@docs
TFFunction
get_param_grad_tf_var
reset_param_grads_tf_op
```

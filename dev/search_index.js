var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#GenTF-1",
    "page": "Home",
    "title": "GenTF",
    "category": "section",
    "text": "TensorFlow plugin for the Gen probabilistic programming systemThe Julia package GenTF allows for Gen generative functions to invoke TensorFlow computations executed on the GPU by the TensorFlow runtime. Users construct a TensorFlow computation using the familiar TensorFlow Python API, and then package the TensorFlow computation in a TFFunction, which is a type of generative function provided by GenTF. Generative functions written in Gen\'s built-in modeling language can seamlessly call TFFunctions. GenTF integrates Gen\'s automatic differentiation with TensorFlow\'s gradients, allowing automatic differentiation of computations that combine Julia and TensorFlow code."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "The installation requires an installation of Python and an installation of the tensorflow Python package. We recommend creating a Python virtual environment and installing TensorFlow via pip in that environment. In what follows, let <python> stand for the absolute path of a Python executable that has access to the tensorflow package.From the Julia REPL, type ] to enter the Pkg REPL mode and run:pkg> add https://github.com/probcomp/GenTFIn a Julia REPL, build the PyCall module so that it will use the correct Python environment:using Pkg; ENV[\"PYTHON\"] = \"<python>\"; Pkg.build(\"PyCall\")Check that intended python environment is indeed being used with:using PyCall; println(PyCall.python)If you encounter problems, see https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version"
},

{
    "location": "#Calling-the-TensorFlow-Python-API-1",
    "page": "Home",
    "title": "Calling the TensorFlow Python API",
    "category": "section",
    "text": "GenTF uses the Julia package PyCall to invoke the TensorFlow Python API.First, import PyCall:using PyCallThen import the tensorflow Python module:@pyimport tensorflow as tfTo import a module from a subpackage:@pyimport tensorflow.train as trainThen, call the TensorFlow Python API with syntax that is very close to Python syntax:W = tf.get_variable(\"W\", dtype=tf.float32, initializer=init_W)\nx = tf.placeholder(tf.float32, shape=(3,), name=\"x\")\ny = tf.squeeze(tf.matmul(W, tf.expand_dims(x, axis=1)), axis=1)\nsess = tf.Session()\nsess[:run](tf.global_variables_initializer())\ny_val = sess[:run](y, feed_dict=Dict(x => [1., 2., 3.]))Here are syntax changes that are required for common situations:Attributes of Python objects (including methods) are accessed using o[:attr] instead of o.attr. Therefore, to run something in a TensorFlow session sess, use sess[:run](..) instead of sess.run(..).\nWhere Python dictionaries would be used, use Julia dictionaries instead.\n[1, 1, 1, 1] constructs a Julia Array, which by default gets converted to a numpy array, not a Julia list. When a TensorFlow API function requires that an argument is a Python list or a tuple (e.g. the strides argument of tf.nn.conv2d), use a Julia tuple: (1, 1, 1, 1).See the PyCall README for the complete description of syntax differences introduced when using Julia and PyCall instead of Python."
},

{
    "location": "#TensorFlow-Generative-Functions-1",
    "page": "Home",
    "title": "TensorFlow Generative Functions",
    "category": "section",
    "text": "A TensorFlow computation graph contains both the model(s) being trained as well as the operations that do the training. In contrast, Gen uses a more rigid separation between models (both generative models and inference models) and the operations that act on models. Specifically, models in Gen are defined as (pure functional) generative functions, and the operations that run the models or train the models are defined in separate Julia code. The GenTF package allows users to construct deterministic generative functions of type TFFunction <: GenerativeFunction from a TensorFlow computation graph in which each TensorFlow element is one of the following roles:Role in TFFunction TensorFlow object type\nArgument tf.Tensor produced by tf.placeholder\nTrainable Parameter tf.Variable\nOperation in Body tf.Tensor produced by non-mutating TensorFlow operation (e.g. tf.conv2d)\nN/A tf.Tensor produced by mutating TensorFlow operation (e.g. tf.assign)TensorFlow placeholders play the role of arguments to the generative function. TensorFlow Variables play the role of the trainable parameters of the generative function. Their value is shared across all invocations of the generative function and is managed by the TensorFlow runtime, not Julia. We will discuss how to train these parameters in section Implementing parameter updates. Tensors produced from non-mutating operations comprise the body of the generative function. One of these elements (either an argument parameter, or element of the body) is designated the return value of the generative function. Note that we do not currently permit TensorFlow generative functions to use randomness.To construct a TensorFlow generative function, we first construct the TensorFlow computation graph using the TensorFlow Python API:using Gen\nusing GenTF\nusing PyCall\n\n@pyimport tensorflow as tf\n@pyimport tensorflow.nn as nn\n\nxs = tf.placeholder(tf.float64) # N x 784\nW = tf.Variable(zeros(Float64, 784, 10))\nb = tf.Variable(zeros(Float64, 10))\nprobs = nn.softmax(tf.add(tf.matmul(xs, W), b), axis=1) # N x 10Then we construct a TFFunction from the TensorFlow graph objects. The first argument to TFFunction is the TensorFlow session, followed by a Vector of trainable parameters (W and b), a Vector of arguments (xs), and finally the return value (probs).sess = tf.Session()\ntf_func = TFFunction([W, b], [xs], probs, sess)The return value must be a differentiable function of each argument and each parameter. Note that the return value does not need to be a scalar. TensorFlow computations for gradients with respect to the arguments and with respect to the parameters are automatically generated when constructing the TFFunction.If a session is not provided a new session is created:tf_func = TFFunction([W, b], [xs], probs)Values for the parameters are managed by the TensorFlow runtime. The TensorFlow session that contains the parameter values is obtained with:sess = get_session(tf_func)The value of a trainable parameter can be obtained in Julia by fetching the Python Variable object (e.g. \'W\'):W_value = sess[:run](W)Equivalently, this can be done using a more concise syntax with the runtf method:W_value = runtf(tf_func, W)"
},

{
    "location": "#What-happens-during-Gen.initialize-1",
    "page": "Home",
    "title": "What happens during Gen.initialize",
    "category": "section",
    "text": "Suppose we run initialize on the TFFunction:(trace, weight) = initialize(tf_func, (xs_val,), EmptyChoiceMap())The TensorFlow runtime computes the return value for the given values of the arguments and the current values of of the trainable parameters.\nThe return value is obtained by Julia from TensorFlow and stored in the trace (it is accessible with get_retval(trace)).\nThe given argument values are also stored in the trace (accessible with get_args(trace)).Note that we pass an empty assignment to initialize because a TFFunction cannot make any random choices that could be constrained."
},

{
    "location": "#What-happens-during-Gen.backprop_trace-1",
    "page": "Home",
    "title": "What happens during Gen.backprop_trace",
    "category": "section",
    "text": "When running backprop_trace with a trace produced from a TFFunction, we must pass a gradient value for the return value. This value should be a Julia Array with the same shape as the return value.((xs_grad,), _, _) = backprop_trace(trace, EmptyAddressSet(), retval_grad)The gradients with respect to each argument are computed by the TensorFlow runtime.\nThe values of the gradient are converted to Julia values and returned.Note that we pass an empty selection because a TFFunction does not make any random choices that could be selected."
},

{
    "location": "#What-happens-during-Gen.backprop_params-1",
    "page": "Home",
    "title": "What happens during Gen.backprop_params",
    "category": "section",
    "text": "When running backprop_params with a trace produced from a TFFunction, we must pass a gradient value for the return value. This value should be a Julia Array with the same shape as the return value.(xs_grad,) = backprop_params(trace, retval_grad)Like backprop_trace, the method returns the value of the gradient with respect to the arguments\nThe gradient with respect to each trainable parameters is computed by the TensorFlow runtime.\nA gradient accumulator TensorFlow Variable for each trainable parameter is incremented by the corresponding gradient value.The gradient accumulator for a parameter accumulates gradient contributions over multiple invocations of backprop_params. A gradient accumulator TensorFlow Variable value can be obtained from the TFFunction with get_param_grad_tf_var (see API below). The value of all gradient accumulators for a given TFFunction can be reset to zeros with reset_param_grads_tf_op (see API below)."
},

{
    "location": "#Implementing-parameter-updates-1",
    "page": "Home",
    "title": "Implementing parameter updates",
    "category": "section",
    "text": "Updates to the trainable parameters of a TFFunction are also defined using the TensorFlow Python API. For example, below we define a TensorFlow operation to apply one step of stochastic gradient descent, based on the current values of the gradient accumulators for all parameters:opt = train.GradientDescentOptimizer(.00001)\ngrads_and_vars = []\npush!(grads_and_vars, (tf.negative(get_param_grad_tf_var(tf_func, W)), W))\npush!(grads_and_vars, (tf.negative(get_param_grad_tf_var(tf_func, b)), b))\nupdate = opt[:apply_gradients](grads_and_vars)We can then apply this update with:sess[:run](update)We can reset the gradient accumulators to zero when desired with:sess[:run](reset_param_grads_tf_op(tf_func))"
},

{
    "location": "#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": "See the examples/ directory for examples that show TFFunctions being combined with regular Gen functions."
},

{
    "location": "#GenTF.TFFunction",
    "page": "Home",
    "title": "GenTF.TFFunction",
    "category": "type",
    "text": "gen_fn = TFFunction(params::Vector{PyObject},\n                    inputs::Vector{PyObject}, output::PyObject,\n                    sess::PyObject=tf.Session())\n\nConstruct a TensorFlow generative function from elements of a TensorFlow computation graph.\n\n\n\n\n\n"
},

{
    "location": "#GenTF.get_session",
    "page": "Home",
    "title": "GenTF.get_session",
    "category": "function",
    "text": "get_session(gen_fn::TFFunction)\n\nReturn the TensorFlow session associated with the given function.\n\n\n\n\n\n"
},

{
    "location": "#GenTF.runtf",
    "page": "Home",
    "title": "GenTF.runtf",
    "category": "function",
    "text": "runtf(gen_fn::TFFunction, ...)\n\nFetch values or run operations in the TensorFlow session associated with the given function.\n\nSyntactic sugar for get_session(gen_fn)[:run](args...)\n\n\n\n\n\n"
},

{
    "location": "#GenTF.get_param_grad_tf_var",
    "page": "Home",
    "title": "GenTF.get_param_grad_tf_var",
    "category": "function",
    "text": "var::PyObject = get_param_grad_tf_var(gen_fn::TFFunction, param::PyObject)\n\nReturn the TensorFlow Variable that stores the gradient of the given parameter TensorFlow Variable.\n\n\n\n\n\n"
},

{
    "location": "#GenTF.reset_param_grads_tf_op",
    "page": "Home",
    "title": "GenTF.reset_param_grads_tf_op",
    "category": "function",
    "text": "op::PyObject = reset_param_grads_tf_op(gen_fn::TFFunction)\n\nReturn the TensorFlow operation Tensor that resets the gradients of all parameters of the given function to zero.\n\n\n\n\n\n"
},

{
    "location": "#API-1",
    "page": "Home",
    "title": "API",
    "category": "section",
    "text": "TFFunction\nget_session\nruntf\nget_param_grad_tf_var\nreset_param_grads_tf_op"
},

]}

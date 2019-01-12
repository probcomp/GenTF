module GenTF

using Gen
using PyCall

const tf = PyNULL()

function __init__()
    copy!(tf, pyimport("tensorflow"))
end

struct TFFunctionTrace
    gen_fn::GenerativeFunction
    args::Tuple
    retval::Any
end

Gen.get_args(trace::TFFunctionTrace) = trace.args
Gen.get_retval(trace::TFFunctionTrace) = trace.retval
Gen.get_assmt(::TFFunctionTrace) = EmptyAssignment()
Gen.get_score(::TFFunctionTrace) = 0.
Gen.get_gen_fn(trace::TFFunctionTrace) = trace.gen_fn

struct TFFunction <: GenerativeFunction{Any,TFFunctionTrace}
    sess::PyObject
    params::Vector{PyObject}
    inputs::Vector{PyObject}
    output::PyObject
    param_grads::Vector{PyObject}
    param_grad_accums::Vector{PyObject}
    input_grads::Vector{PyObject}
    output_grad::PyObject
    param_grad_add_op::PyObject
    accum_zero_op::PyObject
end

function Gen.has_argument_grads(gen_fn::TFFunction)
    ((true for _ in gen_fn.inputs)...,)
end

Gen.accepts_output_grad(gen_fn::TFFunction) = true

function TFFunction(sess, params, inputs, output)
    output_grad = tf[:placeholder](output[:dtype])
    input_grads = tf[:gradients]([output], inputs, [output_grad])
    param_grads = tf[:gradients]([output], params, [output_grad])

    # gradient accumulators
    param_grad_accums = PyObject[tf[:Variable](param) for param in params]

    # the operation that increments the gradient accumulators
    param_grad_add_ops = []
    for (grad, accum) in zip(param_grads, param_grad_accums)
        push!(param_grad_add_ops, tf[:assign_add](accum, grad))
    end
    param_grad_add_op = tf[:group](param_grad_add_ops...)

    # the operation that resets the gradient accumulators to zeros
    accum_zero_ops = []
    for accum in param_grad_accums
        push!(accum_zero_ops, tf[:assign](accum, tf[:zeros_like](accum)))
    end
    accum_zero_op = tf[:group](accum_zero_ops...)

    sess[:run](tf[:global_variables_initializer]())
    sess[:run](accum_zero_op)

    TFFunction(sess, params, inputs, output, param_grads, param_grad_accums, input_grads, output_grad,
        param_grad_add_op, accum_zero_op)
end

function reset_grads!(gen_fn::TFFunction)
    gen_fn.sess[:run](gen_fn.accum_zero_op)
    nothing
end

get_grad_vars(gen_fn::TFFunction) = gen_fn.param_grad_accums

function Gen.initialize(gen_fn::TFFunction, args::Tuple, ::Assignment)
    feed_dict = Dict()
    for (tensor, value) in zip(gen_fn.inputs, args)
        feed_dict[tensor] = value
    end
    retval = gen_fn.sess[:run](gen_fn.output, feed_dict=feed_dict)
    trace = TFFunctionTrace(gen_fn, args, convert(Array{Float64},retval))
    (trace, 0.)
end

function Gen.propose(gen_fn::TFFunction, args::Tuple)
    (trace, _) = initialize(gen_fn, args, EmptyAssignment())
    retval = get_retval(trace)
    (EmptyAssignment(), 0., retval)
end

Gen.project(::TFFunctionTrace, ::AddressSet) = 0.

function Gen.force_update(trace::TFFunctionTrace, ::Tuple, ::Any, ::Assignment)
    (trace, 0., EmptyAssignment(), DefaultRetDiff())
end

function Gen.fix_update(trace::TFFunctionTrace, ::Tuple, ::Any, ::Assignment)
    (trace, 0., EmptyAssignment(), DefaultRetDiff())
end

function Gen.free_update(trace::TFFunctionTrace, ::Tuple, ::Any, ::AddressSet)
    (trace, 0., DefaultRetDiff())
end

function Gen.backprop_trace(trace::TFFunctionTrace, ::AddressSet, retval_grad)
    gen_fn = get_gen_fn(trace)
    args = get_args(trace)
    feed_dict = Dict()
    for (tensor, value) in zip(gen_fn.inputs, args)
        feed_dict[tensor] = value
    end
    feed_dict[gen_fn.output_grad] = retval_grad
    input_grads = gen_fn.sess[:run](gen_fn.input_grads, feed_dict=feed_dict)
    ((input_grads...,), EmptyAssignment(), EmptyAssignment())
end

function Gen.backprop_params(trace::TFFunctionTrace, retval_grad)
    gen_fn = get_gen_fn(trace)
    args = get_args(trace)
    feed_dict = Dict()
    for (tensor, value) in zip(gen_fn.inputs, args)
        feed_dict[tensor] = value
    end
    feed_dict[gen_fn.output_grad] = retval_grad
    result = gen_fn.sess[:run]([gen_fn.input_grads..., gen_fn.param_grad_add_op], feed_dict=feed_dict)
    @assert length(result) == length(gen_fn.input_grads) + 1
    (result[1:end-1]...,)
end

export TFFunction, reset_grads!, get_grad_vars

end # module GenTF

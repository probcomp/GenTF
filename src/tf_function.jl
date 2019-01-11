import Gen
import Gen: @param
using Gen: GenerativeFunction, Assignment, EmptyAssignment, AddressSet, DefaultRetDiff

using TensorFlow
tf = TensorFlow

# TODO allow multiple output tensors (which become a tuple of arrays in Julia)

###################
# TensorFlowTrace #
###################

struct TensorFlowTrace
    gen_fn::GenerativeFunction
    args::Tuple
    retval::Any
end

Gen.get_args(trace::TensorFlowTrace) = trace.args
Gen.get_retval(trace::TensorFlowTrace) = trace.retval
Gen.get_assmt(::TensorFlowTrace) = EmptyAssignment()
Gen.get_score(::TensorFlowTrace) = 0.
Gen.get_gen_fn(trace::TensorFlowTrace) = trace.gen_fn

###########################
# TensorFlow function DSL #
###########################

const inputs = gensym("inputs")
const input_types = gensym("input_types")
const params = gensym("params")
const outputs = gensym("outputs")
const update = gensym("update")

macro input(name, dtype, shape)
    quote
        $(esc(name)) = tf.placeholder($(esc(dtype)); shape=$(esc(shape)), name=String($(QuoteNode(name))))
        push!($(esc(inputs)), $(esc(name)))
        push!($(esc(input_types)), Array{$(esc(dtype)), length($(esc(shape)))})
    end
end

macro param(name::Symbol, initial_value_expr)
    initial_value = gensym("init_value_$name")
    zero_value = gensym("zero_value_$name")
    grad = gensym("grad_variable_$name")
    quote
        $(esc(initial_value)) = $(esc(initial_value_expr))
        $(esc(zero_value)) = zero($(esc(initial_value)))
        $(esc(name)) = tf.Variable($(esc(initial_value)), name=String($(QuoteNode(name))))
        $(esc(grad)) = tf.Variable($(esc(zero_value)), name=String($(QuoteNode(name))) * "_grad")
        $(esc(params))[$(QuoteNode(name))] = ParamDef($(esc(name)), $(esc(grad)), $(esc(zero_value)))
    end
end

macro output(dtype, expr)
    output = gensym("output")
    output_grad = gensym("output_grad")
    quote
        $(esc(output)) = $(esc(expr))
        $(esc(output_grad)) = tf.placeholder($(esc(dtype)))
        push!($(esc(outputs)), ($(esc(output)), $(esc(output_grad))))
        $(esc(output))
    end
end

macro tf_function(expr)
    @assert expr.head == :block
    quote
        # each TensorFlow function has its own graph (and session)
        graph = Graph()
        old_def = TensorFlow.get_def_graph()
        TensorFlow.set_def_graph(graph)
        local $(esc(inputs))
        local $(esc(input_types))
        local $(esc(outputs))
        local $(esc(params))
        try
            $(esc(inputs)) = Tensor[]
            $(esc(input_types)) = Type[]
            $(esc(outputs)) = Tuple{Tensor,Tensor}[]
            $(esc(params)) = Dict{Symbol,ParamDef}()
            $(esc(expr))
        finally
            TensorFlow.set_def_graph(old_def)
        end
        TensorFlowFunction($(esc(inputs)), $(esc(input_types)), $(esc(outputs)),
                           $(esc(params)), graph)
    end
end

struct ParamDef
    value::Variable
    grad::Variable
    zero::Array
end

mutable struct TensorFlowFunction <: Gen.GenerativeFunction{Any,TensorFlowTrace}
    inputs::Vector{Tensor}
    input_types::Vector{Type}
    input_grads::Vector{Tensor}
    output::Tensor
    output_grad::Tensor
    params::Dict{Symbol,ParamDef}
    update_grads::Tensor
    graph::Graph
    session::Union{Session,Nothing}
end

Gen.has_argument_grads(fn::TensorFlowFunction) = (fill(true, length(fn.inputs))...,)
Gen.accepts_output_grad(fn::TensorFlowFunction) = true

function TensorFlowFunction(inputs::Vector{T}, input_types::Vector{Type},
                            outputs::Vector{Tuple{U,V}},
                            params::Dict{Symbol,ParamDef}, graph::Graph) where {T,U,V}
    if length(outputs) != 1
        error("Exactly one output is allowed, but $(length(outputs)) outputs were declared")
    end
    output, output_grad = outputs[1]

    # accumulate parameter gradient
    local input_grads
    local update_grads
    as_default(graph) do
        param_names = collect(keys(params))
        param_vars = [params[name].value for name in param_names]
        param_grads = [params[name].grad for name in param_names]
        param_grad_increments = tf.gradients([output], param_vars, [output_grad])
        update_grads_list = []
        for (grad, grad_increment) in zip(param_grads, param_grad_increments)
            push!(update_grads_list, tf.assign_add(grad, grad_increment))
        end
        update_grads = tf.group(update_grads_list...)

        # input gradient
        input_grads = tf.gradients([output], inputs, [output_grad])
    end

    TensorFlowFunction(inputs, input_types, input_grads,
                       output, output_grad,
                       params, update_grads,
                       graph, nothing)
end

get_inputs(gen_fn::TensorFlowFunction) = gen_fn.inputs
get_input_grads(gen_fn::TensorFlowFunction) = gen_fn.input_grads
get_output(gen_fn::TensorFlowFunction) = gen_fn.output
get_output_grad(gen_fn::TensorFlowFunction) = gen_fn.output_grad
get_param_names(gen_fn::TensorFlowFunction) = keys(gen_fn.params)
get_param_val(gen_fn::TensorFlowFunction, name::Symbol) = gen_fn.params[name].value
get_param_grad(gen_fn::TensorFlowFunction, name::Symbol) = gen_fn.params[name].grad
get_graph(gen_fn::TensorFlowFunction) = gen_fn.graph
get_session(gen_fn::TensorFlowFunction) = gen_fn.session::Session

function init_session!(gen_fn::TensorFlowFunction)
    session = Session(gen_fn.graph)
    gen_fn.session = session
    tf.run(session, tf.global_variables_initializer())
    session
end

function zero_grad(gen_fn::TensorFlowFunction, name::Symbol)
    as_default(get_graph(gen_fn)) do
        tf.assign(gen_fn.params[name].grad, gen_fn.params[name].zero)
    end
end

function exec_tf_function(gen_fn::TensorFlowFunction, args)
    feed_dict = Dict{Tensor,Array{Float32}}()
    for (input, arg) in zip(get_inputs(gen_fn), args)
        feed_dict[input] = arg
    end
    (output_val,) = tf.run(get_session(gen_fn), [get_output(gen_fn)], feed_dict)
    convert(Array{Float64}, output_val)
end

function gradient(gen_fn::TensorFlowFunction, output_grad_val, args)
    feed_dict = Dict{Tensor, Array{Float32}}()
    for (input, arg) in zip(get_inputs(gen_fn), args)
        feed_dict[input] = arg
    end
    feed_dict[get_output_grad(gen_fn)] = convert(Array{Float32},output_grad_val)

    # get the gradient with respect to @inputs
    input_grads = get_input_grads(gen_fn)
    input_grad_vals = tf.run(get_session(gen_fn), input_grads, feed_dict)
    @assert length(input_grad_vals) == length(input_grads)
    
    # update the gradient accumulators for @params
    tf.run(get_session(gen_fn), gen_fn.update_grads, feed_dict)

    # return the input gradient
    map((arr::Array{Float32}) -> convert(Array{Float64}, arr), input_grad_vals)
end


#################################
# generative function interface #
#################################

function check_empty_constraints(constraints)
    if !isempty(constraints)
        error("TensorFlow function got non-empty assignment: $assmt")
    end
end

function Gen.propose(gen_fn::TensorFlowFunction, args)
    retval = exec_tf_function(gen_fn, args)
    (EmptyAssignment(), 0., retval)
end

function Gen.assess(gen_fn::TensorFlowFunction, args, assmt)
    check_empty_constraints(assmt)
    retval = exec_tf_function(gen_fn, args)
    (0., retval)
end

function Gen.initialize(gen_fn::TensorFlowFunction, args::Tuple, constraints::Assignment)
    check_empty_constraints(constraints)
    retval = exec_tf_function(gen_fn, args)
    trace = TensorFlowTrace(gen_fn, args, retval)
    (trace, 0.)
end

Gen.project(::TensorFlowTrace, selection::AddressSet) = 0.

function Gen.force_update(trace::TensorFlowTrace, args::Tuple, argdiff, assmt::Assignment)
    (trace, 0., EmptyAssignment(), DefaultRetDiff())
end

function Gen.fix_update(trace::TensorFlowTrace, args::Tuple, argdiff, assmt::Assignment)
    (trace, 0., EmptyAssignment(), DefaultRetDiff())
end

function Gen.free_update(trace::TensorFlowTrace, args::Tuple, argdiff, selection::AddressSet)
    if !isempty(selection)
        error("TensorFlow function got non-empty selection")
    end
    (trace, 0., DefaultRetDiff())
end

function Gen.backprop_trace(trace::TensorFlowTrace, selection::AddressSet, retval_grad)
    if !isempty(selection)
        error("TensorFlow function got non-empty selection")
    end
    input_grads = gradient(gen_fn, retval_grad, Gen.get_args(trace))
    (input_grads, EmptyAssignment(), EmptyAssignment())
end

function Gen.backprop_params(trace::TensorFlowTrace, retval_grad)
    gen_fn = trace.gen_fn
    input_grads = gradient(gen_fn, retval_grad, Gen.get_args(trace))
    input_grads
end

export @input, @output
export @tf_function
export get_session
export init_session!
export get_param_names
export get_param_val
export get_param_grad
export zero_grad

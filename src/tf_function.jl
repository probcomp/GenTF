using Gen
import Gen: @param, get_param_grad # TODO use the same function names as for GenLite params?

using TensorFlow
tf = TensorFlow

function get_session end

# TODO allow multiple output tensors (which become a tuple of arrays in Julia)

###################
# TensorFlowTrace #
###################

struct TensorFlowTrace
    call::CallRecord{Any}
end

Gen.get_call_record(trace::TensorFlowTrace) = trace.call
Gen.has_choices(trace::TensorFlowTrace) = false
Gen.get_choices(trace::TensorFlowTrace) = Gen.EmptyChoiceTrie()


###########################
# TensorFlow function DSL #
###########################

const inputs = gensym("inputs")
const params = gensym("params")
const outputs = gensym("outputs")
const update = gensym("update")

macro input(name, dtype, shape)
    quote
        $(esc(name)) = tf.placeholder($(esc(dtype)); shape=$(esc(shape)), name=String($(QuoteNode(name))))
        push!($(esc(inputs)), $(esc(name)))
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
    lines = [quote
        $(esc(inputs)) = Tensor[]
        $(esc(outputs)) = Tuple{Tensor,Tensor}[]
        $(esc(params)) = Dict{Symbol,ParamDef}()
    end]
    append!(lines, map(esc, expr.args))
    push!(lines, Expr(:call, :TensorFlowFunction, esc(inputs), esc(outputs), esc(params)))
    Expr(:call, esc(:(tf.as_default)),
        Expr(:(->), Expr(:tuple), Expr(:block, lines...)),
        Expr(:call, esc(:get_def_graph)))#esc(:get_tf_graph)))
end

struct ParamDef
    value::Variable
    grad::Variable
    zero::Array
end

struct TensorFlowFunction <: Gen.Generator{Any,TensorFlowTrace}
    inputs::Vector{Tensor}
    input_grads::Vector{Tensor}
    output::Tensor
    output_grad::Tensor
    params::Dict{Symbol,ParamDef}
    update_grads::Tensor
end

Gen.accepts_output_grad(::TensorFlowFunction) = true
Gen.has_argument_grads(fn::TensorFlowFunction) = (fill(true, length(fn.inputs))...,)

function TensorFlowFunction(inputs::Vector{T}, outputs::Vector{Tuple{U,V}},
                            params::Dict{Symbol,ParamDef}) where {T,U,V}
    if length(outputs) != 1
        error("Exactly one output is allowed")
    end
    output, output_grad = outputs[1]

    # accumulate parameter gradient
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

    TensorFlowFunction(inputs, input_grads, output, output_grad, params, update_grads)
end

get_inputs(tf_func::TensorFlowFunction) = tf_func.inputs
get_input_grads(tf_func::TensorFlowFunction) = tf_func.input_grads
get_output(tf_func::TensorFlowFunction) = tf_func.output
get_output_grad(tf_func::TensorFlowFunction) = tf_func.output_grad
get_param_names(tf_func::TensorFlowFunction) = keys(tf_func.params)
get_param_val(tf_func::TensorFlowFunction, name::Symbol) = tf_func.params[name].value
get_param_grad(tf_func::TensorFlowFunction, name::Symbol) = tf_func.params[name].grad

function zero_grad(tf_func::TensorFlowFunction, name::Symbol)
    tf.assign(tf_func.params[name].grad, tf_func.params[name].zero)
end

function exec_tf_function(tf_func::TensorFlowFunction, args)
    feed_dict = Dict{Tensor,Array{Float32}}()
    for (input, arg) in zip(get_inputs(tf_func), args)
        feed_dict[input] = arg
    end
    (output_val,) = run(get_session(), [get_output(tf_func)], feed_dict)
    convert(Array{Float64}, output_val)
end

function grad(tf_func::TensorFlowFunction, output_grad_val, args)
    feed_dict = Dict{Tensor, Array{Float32}}()
    for (input, arg) in zip(get_inputs(tf_func), args)
        feed_dict[input] = arg
    end
    feed_dict[get_output_grad(tf_func)] = convert(Array{Float32},output_grad_val)

    # get the gradient with respect to @inputs
    input_grads = get_input_grads(tf_func)
    input_grad_vals = run(get_session(), input_grads, feed_dict)
    @assert length(input_grad_vals) == length(input_grads)
    
    # update the gradient accumulators for @params
    run(get_session(), tf_func.update_grads, feed_dict)

    # return the input gradient
    map((arr::Array{Float32}) -> convert(Array{Float64}, arr), input_grad_vals)
end


############
# simulate #
############

function Gen.simulate(tf_func::TensorFlowFunction, args, read_trace)
    retval = exec_tf_function(tf_func, args)
    trace = TensorFlowTrace(Gen.CallRecord{Any}(0., retval, args))
    trace
end


############
# generate #
############

function check_empty_constraints(constraints)
    if !isempty(constraints)
        error("Attempted to constrain random choice(s) that do not exist")
    end
end

function Gen.generate(fn_func::TensorFlowFunction, args, constraints, read_trace)
    check_empty_constraints(constraints)
    retval = exec_tf_function(tf_func, args)
    trace = TensorFlowTrace(CallRecord{Any}(0., retval, args))
    (trace, 0.)
end


###########
# project #
###########

function Gen.project(tf_func::TensorFlowFunction, args, constraints, read_trace=nothing)
    check_empty_constraints(constraints)
    retval = exec_tf_function(tf_func, args)
    trace = TensorFlowTrace(CallRecord{Any}(0., retval, args))
    (trace, EmptyChoiceTrie())
end


###################
# backprop_params #
###################

function Gen.backprop_params(tf_func::TensorFlowFunction, trace::TensorFlowTrace, retval_grad, read_trace=nothing)
    call = get_call_record(trace)
    input_grads = grad(tf_func, retval_grad, call.args)
    input_grads
end

export @input, @output
export @tf_function
export TensorFlowFunction
export get_param_names
export get_param_val
export get_param_grad
export zero_grad

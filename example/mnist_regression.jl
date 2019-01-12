##############
# load mnist #
##############

import Random
Random.seed!(1)

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata() 

mutable struct DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, Random.shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float64, batch_size, 784)
    y = Vector{Int}(undef, batch_size)
    for i=1:batch_size
        x[i, :] = reshape(train_x[:,:,loader.cur_id], (28*28))
        y[i] = train_y[loader.cur_id] + 1
        loader.cur_id += 1
        if loader.cur_id > 60000
            loader.cur_id = 1
        end
    end
    x, y
end

function load_test_set()
    test_x, test_y = MLDatasets.MNIST.testdata()
    N = length(test_y)
    x = zeros(Float64, N, 784)
    y = Vector{Int}(undef, N)
    for i=1:N
        x[i, :] = reshape(test_x[:,:,i], (28*28))
        y[i] = test_y[i]+1
    end
    x, y
end

const loader = DataLoader()

(test_x, test_y) = load_test_set()


################
# define model #
################

using Gen
using GenTF
using PyCall

@pyimport tensorflow as tf
@pyimport tensorflow.nn as nn

xs = tf.placeholder(tf.float64) # N x 784
W = tf.Variable(zeros(Float64, 784, 10))
b = tf.Variable(zeros(Float64, 10))
probs = nn.softmax(tf.add(tf.matmul(xs, W), b), axis=1) # N x 10

const sess = tf.Session()
const net = TFFunction(sess, [W, b], [xs], probs)

@gen function f(xs::Matrix{Float64})
    (N, D) = size(xs)
    @assert D == 784
    probs = @addr(net(xs), :net)
    @assert size(probs) == (N, 10)
    ys = Vector{Int}(undef, N)
    for i=1:N
        #normed = probs[i,:] / sum(probs[i,:])
        ys[i] = @addr(categorical(probs[i,:]), (:y, i))
    end
    return ys
end


#########
# train #
#########

@pyimport tensorflow.train as train

function make_update()
    opt = train.GradientDescentOptimizer(.00001)
    grads_and_vars = []
    push!(grads_and_vars, (tf.negative(get_param_grad_tf_var(net, W)), W))
    push!(grads_and_vars, (tf.negative(get_param_grad_tf_var(net, b)), b))
    sgd_step = opt[:apply_gradients](grads_and_vars)
    update = nothing
    @pywith tf.control_dependencies([sgd_step]) begin
        update = tf.group(reset_param_grads_tf_op(net))
    end
    update
end

const update = make_update()

for i=1:100000

    (xs, ys) = next_batch(loader, 100)

    @assert size(xs) == (100, 784)
    @assert size(ys) == (100,)
    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints[(:y, i)] = y
    end

    (trace, weight) = initialize(f, (xs,), constraints)

    # increments gradient accumulators
    backprop_params(trace, nothing)

    # performs SGD update and then resets gradient accumulators
    sess[:run](update)

    println("i: $i, weight: $weight")
end

##################################
# sample inferences on test data #
##################################

for i=1:length(test_y[1:100])
    x = test_x[i:i,:]
    @assert size(x) == (1, 784)
    true_y = test_y[i]
    pred_y = f(x)
    @assert length(pred_y) == 1
    println("true: $true_y, predicted: $(pred_y[1])")
end

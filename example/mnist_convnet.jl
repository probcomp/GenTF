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

function weight_variable(shape)
    initial = 0.001 * randn(shape...)
    tf.Variable(initial)
end

function bias_variable(shape)
    initial = fill(.1, shape...)
    tf.Variable(initial)
end

function conv2d(x, W)
    nn.conv2d(x, W, (1, 1, 1, 1), "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), "SAME")
end

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(xs, [-1, 28, 28, 1])

h_conv1 = nn.relu(tf.add(conv2d(x_image, W_conv1), b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = nn.relu(tf.add(conv2d(h_pool1, W_conv2), b_conv2))
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

probs = nn.softmax(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2), axis=1) # N x 10

const sess = tf.Session()
const params = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
const net = TFFunction(sess, params, [xs], probs)

@gen function f(xs::Matrix{Float64})
    (N, D) = size(xs)
    @assert D == 784
    probs = @addr(net(xs), :net)
    @assert size(probs) == (N, 10)
    ys = Vector{Int}(undef, N)
    for i=1:N
        ys[i] = @addr(categorical(probs[i,:]), (:y, i))
    end
    return ys
end


#########
# train #
#########

@pyimport tensorflow.train as train

function make_update()
    opt = train.AdamOptimizer(1e-4)
    grads_and_vars = []
    for param in params
        grad = tf.negative(get_param_grad_tf_var(net, param))
        push!(grads_and_vars, (grad, param))
    end
    sgd_step = opt[:apply_gradients](grads_and_vars)
    update = nothing
    @pywith tf.control_dependencies([sgd_step]) begin
        update = tf.group(reset_param_grads_tf_op(net))
    end
    update
end

const update = make_update()

sess[:run](tf.global_variables_initializer())

for i=1:10000

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

for i=1:length(test_y)
    x = test_x[i:i,:]
    @assert size(x) == (1, 784)
    true_y = test_y[i]
    pred_y = f(x)
    @assert length(pred_y) == 1
    println("true: $true_y, predicted: $(pred_y[1])")
end

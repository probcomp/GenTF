using Gen 
using GenTF

using TensorFlow
tf = TensorFlow

function strip_lineinfo(expr::Expr) 
    @assert !(expr.head == :line)
    new_args = []
    for arg in expr.args
        if (isa(arg, Expr) && arg.head == :line) || isa(arg, LineNumberNode)
        elseif isa(arg, Expr) && arg.head == :block
            stripped = strip_lineinfo(arg)
            append!(new_args, stripped.args)
        else
            push!(new_args, strip_lineinfo(arg))
        end
    end
    Expr(expr.head, new_args...)
end

strip_lineinfo(expr) = expr

using Cairo
using FileIO
using ImageMagick
using Compat
using Compat.Base64
using Images, ImageView
using ImageFiltering

const letters = String["A", "B", "C"]

###########################
# primitive distributions #
###########################

struct NoisyMatrix <: Distribution{Matrix{Float64}} end
const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
        end
    end
    mat
end

####################
# generative model #
####################

struct Object
    x::Float64
    y::Float64
    angle::Float64
    fontsize::Int
    letter::String
end

const width = 56
const height = 56
const min_size = 15
const max_size = 35

function render(obj::Object)
    canvas = CairoRGBSurface(width, height)
    cr = CairoContext(canvas)
    Cairo.save(cr)

    # set background color to white
    set_source_rgb(cr, 1.0, 1.0, 1.0)
    rectangle(cr, 0.0, 0.0, width, height)
    fill(cr)
    restore(cr)
    Cairo.save(cr)

    # write some letters
    set_font_face(cr, "Sans $(obj.fontsize)")
    text(cr, obj.x, obj.y, obj.letter, angle=obj.angle)

    # convert to matrix of color values
    buf = IOBuffer()
    write_to_png(canvas, buf)
    Images.Gray.(readblob(take!(buf)))
end

size_to_int(size) = min_size + Int(floor((max_size - min_size + 1) * size))

@gen function model()

    # object prior
    x = @addr(uniform_continuous(0, 1), "x")
    y = @addr(uniform_continuous(0, 1), "y")
    size = @addr(uniform_continuous(0, 1), "size")
    letter = letters[@addr(uniform_discrete(1, length(letters)), "letter")]
    angle = @addr(uniform_continuous(-1, 1), "angle")
    fontsize = size_to_int(size)
    object = Object(height * x, width * y, angle * 45, fontsize, letter)

    # render
    image = render(object)

    # blur it
    blur_amount = 3
    blurred = imfilter(image, Kernel.gaussian(blur_amount))

    # add speckle
    mat = convert(Matrix{Float64}, blurred)
    noise = 0.1
    @addr(noisy_matrix(mat, noise), "image")
end

function render_trace(trace, fname::String)
    x = height * trace["x"]
    y = width * trace["y"]
    angle = trace["angle"] * 45
    fontsize = size_to_int(trace["size"])
    letter = letters[trace["letter"]]
    object = Object(x, y, angle, fontsize, letter)
    image = render(object)
    image = min.(fill(1, size(image)), max.(zero(image), image))
    FileIO.save(fname, colorview(Gray, image))
end

####################################
# inference network based proposal #
####################################

const num_input = width * height
const num_output = 11

function conv2d(x, W)
    tf.nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

function max_pool_2x2(x)
    tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

function initial_weight(shape)
    randn(Float32, shape...) * 0.001f0
end

function initial_bias(shape)
    fill(0.1f0, shape...)
end


inference_network = @tf_function begin

    # input image
    @input image_flat Float32 [-1, num_input]
    image = tf.reshape(image_flat, [-1, width, height, 1])

    # convolution + max-pooling (28 x 28 x 32)
    @param W_conv1 initial_weight([5, 5, 1, 32])
    @param b_conv1 initial_bias([32])
    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # convolution + max-pooling (14 x 14 x 32)
    @param W_conv2 initial_weight([5, 5, 32, 32])
    @param b_conv2 initial_bias([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 14 * 14 * 32])

    # convolution + max-pooling (7 x 7 x 64)
    @param W_conv3 initial_weight([5, 5, 32, 64])
    @param b_conv3 initial_bias([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 64])

    # fully connected layer
    @param W_fc1 initial_weight([7 * 7 * 64, 1024])
    @param b_fc1 initial_bias([1024])
    h_fc1 = tf.nn.relu(h_pool3_flat * W_fc1 + b_fc1)

    # output layer
    @param W_fc2 initial_weight([1024, num_output])
    @param b_fc2 initial_bias([num_output])
    @output Float32 (tf.matmul(h_fc1, W_fc2) + b_fc2)
end

function make_inference_network_update()
    net = inference_network

    # get accumulated negative gradients of log probability with respect to each parameter
    grads_and_vars = [
        (tf.negative(get_param_grad(net, n)), get_param_val(net, n)) for n in get_param_names(net)]

    # use ADAM 
    optimizer = tf.train.AdamOptimizer(1e-4)

    tf.group(
        tf.train.apply_gradients(optimizer, grads_and_vars),
        [zero_grad(net, n) for n in get_param_names(net)]...)
end

inference_network_update = make_inference_network_update()

@gen function dl_proposal_predict(@ad(outputs))

    # predict the x-coordinate
    x_mu = outputs[1]
    x_std = exp.(outputs[2])
    @addr(normal(x_mu, x_std), "x")

    # predict the y-coordinate
    y_mu = outputs[3]
    y_std = exp.(outputs[4])
    @addr(normal(y_mu, y_std), "y")

    # predict the rotation
    r_mu = exp.(outputs[5])
    r_std = exp.(outputs[6])
    @addr(normal(r_mu, r_std), "angle")

    # predict the size 
    size_alpha = exp(outputs[7])
    size_beta = exp(outputs[8])
    @addr(Gen.beta(size_alpha, size_beta), "size")
    
    # predict the identity of the letter
    log_letter_dist = outputs[9:9 + length(letters)-1]
    letter_dist = exp.(log_letter_dist)
    letter_dist = letter_dist / sum(letter_dist)
    @addr(categorical(letter_dist), "letter")
end

@gen function dl_proposal()

    # get image from input trace
    image_flat = zeros(1, num_input)
    image_flat[1,:] = @read("image")[:]

    # run inference network
    outputs = @addr(inference_network(image_flat), :network)

    # make prediction given inference network outputs
    @splice(dl_proposal_predict(outputs[1,:]))
end

@gen function dl_proposal_batched(batch_size::Int)

    # get images from input trace
    images_flat = zeros(Float32, batch_size, width * height)
    for i=1:batch_size
        images_flat[i,:] = @read(i => "image")[:]
    end

    # run inference network in batch
    outputs = @addr(inference_network(images_flat), :network)
    
    # make prediction for each image given inference network outputs
    for i=1:batch_size
        @addr(dl_proposal_predict(outputs[i,:]), :predictions => i)
    end
end

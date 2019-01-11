using Gen
using GenTF
using TensorFlow
tf = TensorFlow
using Test

@testset "basic" begin

    init_W = rand(Float32, 2, 3)
    foo = @tf_function begin
        @input x Float32 [3]
        @param W init_W
        y = tf.dropdims(tf.matmul(W, tf.expand_dims(x, 2)), dims=[2])
        @output Float32 y
    end
    
    foo_session = init_session!(foo)

    x = rand(Float32, 3)
    (trace, weight) = initialize(foo, (x,), EmptyAssignment())
    @test weight == 0.
    y = get_retval(trace)
    @test isapprox(y, init_W * x)
    y_grad = rand(Float32, 2)

    (x_grad,) = backprop_params(trace, y_grad)
    @test isapprox(x_grad, init_W' * y_grad)

    W_grad = tf.run(foo_session, GenTF.get_param_grad(foo, :W))
    @test isapprox(W_grad, y_grad * x')
end

@testset "maximum likelihood" begin

    N = 4

    tf_func = @tf_function begin
        @input xs Float32 [N] # (N,)
        @param w Float32[0., 0.] # (2,)
        ones = tf.fill(tf.constant(1.0, dtype=Float32), [N])
        X = tf.stack([xs, ones], axis=2) # (N,2)
        y_means = tf.dropdims(tf.matmul(X, tf.expand_dims(w, 2)), dims=[2]) # (N,)
        @output Float32 y_means
    end

    @gen function model(xs::Vector{Float64})
        y_means = @addr(tf_func(xs), :tf_func)
        for i=1:length(xs)
            @addr(normal(y_means[i], 1.), "y-$i")
        end
    end

    update = tf.as_default(GenTF.get_graph(tf_func)) do 
        w_var = get_param_val(tf_func, :w)
        w_grad = GenTF.get_param_grad(tf_func, :w)
        gradient_step = tf.assign_add(w_var, tf.mul(w_grad, tf.constant(0.01, dtype=Float32)))
        tf.with_op_control([gradient_step]) do
            zero_grad(tf_func, :w)
        end
    end

    tf_func_session = init_session!(tf_func)

    xs = Float64[-2, -1, 1, 2]
    ys = -2 * xs .+ 1
    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints["y-$i"] = y
    end
    for iter=1:200
        (trace, _) = initialize(model, (xs,), constraints)
        score = get_score(trace)
        w = tf.run(tf_func_session, get_param_val(tf_func, :w))
        backprop_params(trace, nothing)
        w_grad = tf.run(tf_func_session, GenTF.get_param_grad(tf_func, :w))
        tf.run(tf_func_session, update)
    end
    w = tf.run(tf_func_session, get_param_val(tf_func, :w))
    @test isapprox(w[1], -2., atol=0.001)
    @test isapprox(w[2], 1., atol=0.01)
    
end

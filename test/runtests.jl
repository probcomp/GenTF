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
        y = tf.squeeze(tf.matmul(W, tf.expand_dims(x, 2)))
        @output Float32 y
    end
    
    Gen.load_generated_functions()
    session = Session(get_def_graph())
    GenTF.get_session() = session
    tf.run(session, tf.global_variables_initializer())

    x = rand(Float32, 3)
    trace = assess(foo, (x,), EmptyChoiceTrie())
    y = get_call_record(trace).retval
    @test isapprox(y, init_W * x)
    y_grad = rand(Float32, 2)
    (x_grad,) = backprop_params(foo, trace, y_grad)
    @test isapprox(x_grad, init_W' * y_grad)
    W_grad = tf.run(session, get_param_grad(foo, :W))
    @test isapprox(W_grad, y_grad * x')
end

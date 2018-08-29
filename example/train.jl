import FileIO
import Random
using Printf

include("model.jl")

Gen.load_generated_functions()

function generate_prior_samples()
    for i=1:100
        trace = simulate(model, ())
        img = get_call_record(trace).retval
        img = max.(zero(img), img)
        img = min.(fill(1., size(img)), img)
        id = @sprintf("%03d", i)
        FileIO.save("prior/img-$id.png", colorview(Gray, img))
    end
end
println("generating prior samples..")
generate_prior_samples()

const num_train = 100  

function generate_training_data()
    traces = Vector{Any}(undef, num_train)
    for i=1:num_train
        traces[i] = get_choices(simulate(model, ()))
        @assert !isempty(traces[i])
        if i % 100 == 0
            println("$i of $num_train")
        end
    end
    traces
end

function initial_weight(shape)
    randn(Float32, shape...) * 0.001f0
end

function initial_bias(shape)
    fill(0.1f0, shape...)
end

session = Session(get_def_graph())
GenTF.get_session() = session

function train_inference_network(all_traces, num_iter)

    # do training
    batch_size = 100
    for iter=1:num_iter
        minibatch = Random.randperm(num_train)[1:batch_size]
        traces = all_traces[minibatch]
        vector_trace = vectorize_internal(traces)
        constraints = DynamicChoiceTrie()
        set_internal_node!(constraints, :predictions, vector_trace)
        (batched_trace, _) = project(dl_proposal_batched, (batch_size,), constraints, Some(vector_trace))
        score = get_call_record(batched_trace).score / batch_size
        backprop_params(dl_proposal_batched, batched_trace, nothing, Some(vector_trace))
        tf.run(session, inference_network_update)
        println("iter: $iter, score: $(score)")
        if iter % 10 == 0
            saver = tf.train.Saver()
            println("saving params...")
            tf.train.save(saver, session, "infer_net_params.jld2")
        end
    end
end

println("generating training data...")
const traces = generate_training_data()

tf.run(session, tf.global_variables_initializer())

saver = tf.train.Saver()
tf.train.restore(saver, session, "infer_net_params.jld2")

println("training...")
train_inference_network(traces, 100)

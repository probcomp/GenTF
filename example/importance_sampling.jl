import FileIO
import Random
using Printf

include("model.jl")

import TensorFlow
tf = TensorFlow
Gen.load_generated_functions()

function logsumexp(arr::Vector{Float64})
    maxlog = maximum(arr)
    maxlog + log(sum(exp.(arr .- maxlog)))
end

session = Session(get_def_graph())
GenTF.get_session() = session

function do_inference(input_image, n::Int)

    observations = DynamicChoiceTrie()
    observations["image"] = input_image

    # do importance sampling
    traces = Vector{Any}(n)
    log_weights = Vector{Float64}(n)
    for i=1:n
        latents = get_choices(simulate(dl_proposal, (observations,)))
        constraints = merge!(observations, latents)
        (trace, log_weights[i]) = generate(model, (), constraints)
        traces[i] = get_choices(trace)
    end
    dist = exp.(log_weights .- logsumexp(log_weights))
    idx = random(categorical, dist)
    traces[idx]
end

tf.run(session, tf.global_variables_initializer())
saver = tf.train.Saver()
tf.train.restore(saver, session, "infer_net_params.jld2")

input_fname = "prior/img-002.png"
input_image = convert(Matrix{Float64}, load(input_fname))
for n in [10]#, 10, 100, 1000]
    reps = 20
    runtimes = Vector{Float64}(reps)
    for i=1:reps
        start = time()
        predicted = do_inference(input_image, n)
        runtimes[i] = time() - start
        output_fname = @sprintf("importance-sampling/img-002/n%03d.%03d.png", n, i)
        render_trace(predicted, output_fname)
    end
    println("n: $n, median runtime (sec.): $(median(runtimes))")
end

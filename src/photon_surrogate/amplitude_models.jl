export AbsScaPoissonExpModelParams, PoissonExpModelParams, AbsScaPoissonExpFourierModelParams
export GNNAmplitudeModel
export SimpleMLPAmpModel, SimpleMLPAmpModelParams
export PhotonSurrogateAmplitude

export create_model_input

struct PhotonSurrogateAmplitude{A<:AmplitudeSurrogate} <: PhotonSurrogate
    model::A
end

function PhotonSurrogateAmplitude(fname_amp)

    amp_model = load(fname_amp)[:model]
    Flux.testmode!(amp_model)

    return PhotonSurrogateAmplitude(amp_model)
end

function create_input_buffer(model::PhotonSurrogateAmplitude, n_det::Integer, max_particles=500)
    input_size = size(model.model.embedding.layers[1].weight, 2)
    return create_input_buffer(input_size, n_det, max_particles)
end



struct PoissonExpModel <: AmplitudeSurrogate
    embedding::Chain
    transformations::Vector{<:Transformation}
end

# Make embedding parameters trainable
Flux.@layer PoissonExpModel trainable=(embedding,)

function (m::PoissonExpModel)(cond)
    output = m.embedding(cond)
    return output
end


struct PoissonExpFourierModel{C, T, E} <: AmplitudeSurrogate
    embedding::C
    transformations::T
    embedding_matrix::E
end

Flux.@layer PoissonExpFourierModel trainable=(embedding,)

function (m::PoissonExpFourierModel)(x)

    # First apply transformations
    x_transf = apply_transformation(x, m.transformations)

    # Apply the embedding matrix
    x_embed = fourier_input_mapping(x_transf, m.embedding_matrix)

    output = m.embedding(x_embed)
    return output
end

Base.@kwdef struct AbsScaPoissonExpFourierModelParams <: HyperParams
    batch_size::Int64 = 16384
    mlp_layers::Int64 = 3
    mlp_layer_size::Int64 = 768
    lr::Float64 = 0.0023
    lr_min::Float64 = 1E-8
    epochs::Int64 = 150
    dropout::Float64 = 0.0044
    non_linearity::String = "gelu"
    seed::Int64 = 31338
    l2_norm_alpha = 0.1
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    resnet = false
    fourier_gaussian_scale = 0.1
    fourier_mapping_size = 64
end


Base.@kwdef struct AbsScaPoissonExpModelParams <: HyperParams
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    lr_min::Float64 = 1E-5
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::String = "relu"
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    resnet = false
end

Base.@kwdef struct PoissonExpModelParams <: HyperParams
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    lr_min::Float64 = 1E-5
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::String = "relu"
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    resnet = false
end

"""
    setup_model(hparams::PoissonExpModel)

Constructs a model for photon amplitude prediction using a Poisson model.

# Arguments
- `hparams::PoissonExpModelParams`: Hyperparameters for the model.
- `transformations::AbstractVector{<:Transformation}`: Feature Transformations.

# Returns
- `embedding`: The embedding layer of the model.
- `log_poisson_likelihood`: The log-likelihood function for the Poisson distribution.

"""
function setup_model(hparams::PoissonExpModelParams, transformations::AbstractVector{<:Transformation})
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lin = NON_LINS[hparams.non_linearity]

    n_in = 8 
    n_out = 16

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    return PoissonExpModel(embedding, transformations), log_poisson_likelihood
    
end

"""
    setup_model(hparams::AbsScaPoissonExpModelParams)

    Constructs a model for photon amplitude prediction using a Poisson model with medium perturbations.

# Arguments
- `hparams::AbsScaPoissonExpModel`: The hyperparameters for the model.
- `transformations::AbstractVector{<:Normalizer}`: Feature normalizers.

# Returns
- `embedding`: The embedding layer of the model.
- `log_poisson_likelihood`: The log Poisson likelihood function.

"""
function setup_model(hparams::AbsScaPoissonExpModelParams, transformations::AbstractVector{<:Transformation})
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lin = NON_LINS[hparams.non_linearity]

    n_in = 10 
    n_out = 16

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    
    return PoissonExpModel(embedding, transformations), log_poisson_likelihood
end



"""
    setup_model(hparams::AbsScaPoissonExpFourierModelParams)

    Constructs a model for photon amplitude prediction using a Poisson model with medium perturbations.

# Arguments
- `hparams::AbsScaPoissonExpFourierModelParams`: The hyperparameters for the model.
- `transformations::AbstractVector{<:Transformation}`: Feature Transformations.
- `embedding_matrix::Matrix`: The embedding matrix.

# Returns
- `embedding`: The embedding layer of the model.
- `log_poisson_likelihood`: The log Poisson likelihood function.

"""
function setup_model(hparams::AbsScaPoissonExpFourierModelParams, transformations::AbstractVector{<:Transformation}, embedding_matrix::Matrix)

    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    n_in = 2*hparams.fourier_mapping_size
    n_out = 16

    non_lin = NON_LINS[hparams.non_linearity]

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    
    return PoissonExpFourierModel(embedding, transformations, embedding_matrix), log_poisson_likelihood
end

"""
    setup_model(hparams::AbsScaPoissonExpFourierModelParams)

    Constructs a model for photon amplitude prediction using a Poisson model with medium perturbations.

# Arguments
- `hparams::AbsScaPoissonExpFourierModelParams`: The hyperparameters for the model.
- `transformations::AbstractVector{<:Transformation}`: Feature Transformations.

# Returns
- `embedding`: The embedding layer of the model.
- `log_poisson_likelihood`: The log Poisson likelihood function.

"""
function setup_model(hparams::AbsScaPoissonExpFourierModelParams, transformations::AbstractVector{<:Transformation})
    rand_mat = randn(Float32, (hparams.fourier_mapping_size, 10))  * hparams.fourier_gaussian_scale
    return setup_model(hparams, transformations, rand_mat)
end


function create_model_input!(
    model::PoissonExpModel,
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    output;
    abs_scale,
    sca_scale)
   return create_model_input!(particles, targets, output, model.transformations, abs_scale=abs_scale, sca_scale=sca_scale)
end

function create_model_input!(
    model::PoissonExpModel,
    target_particles_vector::Vector{<:Tuple{<:PhotonTarget, Vector{Particle}}},
    output;
    abs_scale,
    sca_scale)
   return create_model_input!(target_particles_vector, output, model.transformations, abs_scale=abs_scale, sca_scale=sca_scale)
end



Base.@kwdef struct SimpleMLPAmpModelParams <: HyperParams
    n_features::Int64 = 6
    n_hidden::Int64 = 256
    dropout::Float64 = 0.0
    non_linearity::String = "gelu"
    l2_norm_alpha = 0.0
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    use_skip = false
end

struct SimpleMLPAmpModel <: AmplitudeSurrogate
    model::Chain
end

Flux.@layer SimpleMLPAmpModel trainable=(model,)

function SimpleMLPAmpModel(;n_features=6, n_hidden=1024, non_lin="relu", dropout=0.0, use_skip=false)
    non_lin_lookup = Dict("relu" => NNlib.relu, "tanh" => NNlib.tanh, "sigmoid" => NNlib.sigmoid, "celu" => NNlib.celu, "gelu" => NNlib.gelu, "swish" => NNlib.swish, "selu" => NNlib.selu)
    non_lin = non_lin_lookup[non_lin]
    if use_skip
        model = Chain(
            Dense(n_features, n_hidden, non_lin),
            Dropout(dropout),
            SkipConnection(Dense(n_hidden, n_hidden, non_lin), +),
            Dropout(dropout),
            SkipConnection(Dense(n_hidden, n_hidden, non_lin), +),
            Dropout(dropout),
            Dense(n_hidden, 16),
        )
    else
        model = Chain(
            Dense(n_features, n_hidden, non_lin),
            Dropout(dropout),
            Dense(n_hidden, n_hidden, non_lin),
            Dropout(dropout),
            Dense(n_hidden, n_hidden, non_lin),
            Dropout(dropout),
            Dense(n_hidden, 16),
        )
    end
    return SimpleMLPAmpModel(model)
end

function SimpleMLPAmpModel(hparams::SimpleMLPAmpModelParams)
    return SimpleMLPAmpModel(n_features=hparams.n_features, n_hidden=hparams.n_hidden, non_lin=hparams.non_linearity)
end

# Define a loss function (example: mean squared error)
function loss(model::SimpleMLPAmpModel, batch)

    preds = model(batch.x)
    targets = batch.nhits
    return Flux.mse(preds, targets)
end

function (m::SimpleMLPAmpModel)(x)
    return m.model(x)
end

function (m::SimpleMLPAmpModel)(target_position, particle_pos, particle_dir, particle_energy)
    input = collect(create_model_input(m, particle_pos, particle_dir, particle_energy, target_position))
    return m.model(input)
end


function create_model_input(::SimpleMLPAmpModel, particle_pos, particle_dir, particle_energy, target_pos)
    rel_pos = particle_pos - target_pos
    dist = norm(rel_pos)

    pos_theta, pos_phi = cart_to_sph(rel_pos ./ dist)
    dir_theta, dir_phi = cart_to_sph(particle_dir)

    return log10(dist), log10(particle_energy), cos(pos_theta), pos_phi, cos(dir_theta), dir_phi
end


function create_model_input(model::SimpleMLPAmpModel, particle::Particle, target::PhotonTarget)
    return create_model_input(model, particle.position, particle.direction, particle.energy, target.shape.position)
end


function create_model_input!(
    model::SimpleMLPAmpModel,
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    output;)
   
    out_ix = LinearIndices((eachindex(particles), eachindex(targets)))

    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        ix = out_ix[p_ix, t_ix]

        outview = @view output[:, ix]

        model_input = create_model_input(model, particle, target)
        outview .= model_input
    end
end

function get_log_amplitudes(particles, targets, model::SimpleMLPAmpModel; feat_buffer, device=gpu)

    n_pmt = get_pmt_count(eltype(targets))

    #TODO: get rid of this
    input_size = size(model.model.layers[1].weight, 2)

    amp_buffer = @view feat_buffer[1:input_size, 1:length(targets)*length(particles)]
    create_model_input!(model, particles, targets, amp_buffer)

    input = amp_buffer
    input = permutedims(input)'

    log_expec_per_src_trg::Matrix{eltype(input)} = cpu(model.model(device(input)))

    log_expec_per_src_pmt_rs = reshape(
        log_expec_per_src_trg,
        n_pmt, length(particles), length(targets))

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_pmt_rs, dims=2)

    return log_expec_per_pmt, log_expec_per_src_pmt_rs
end



#=

struct GNNAmplitudeModel <: AmplitudeSurrogate
    node_mapping::Chain
    gnn_model::GNNChain
    n_features::Int64
    node_embedding_dim::Int64
end

function GNNAmplitudeModel(;
    n_features=6,
    node_embedding_dim=64,
    n_nodes=16)

    node_mapping = Chain(
        Dense(n_features, n_nodes * node_embedding_dim, NNlib.relu),
        Dense(n_nodes * node_embedding_dim, n_nodes * node_embedding_dim, NNlib.relu),
        Dense(n_nodes * node_embedding_dim, n_nodes * node_embedding_dim, NNlib.relu),
    )

    gnn_model = GNNChain(
        GCNConv(node_embedding_dim => 1, NNlib.relu, add_self_loops=true),
        #GCNConv(gnn_conv_dim => 1, NNlib.relu,add_self_loops=true),
    )

    return GNNAmplitudeModel(node_mapping, gnn_model, n_features, node_embedding_dim)


end
Flux.@layer GNNAmplitudeModel

function (m::GNNAmplitudeModel)(batched_graph)
    global_input = reshape(batched_graph.x, m.n_features, :)

    mapped = m.node_mapping(global_input)
    # batch dimension is last dimension (n_feat * n_graphs)
    # reshape features to (n_feat, n_nodes * n_graphs)
    mapped = reshape(mapped, m.node_embedding_dim, :)

    pred = m.gnn_model(batched_graph, mapped)


    return pred
end
=#
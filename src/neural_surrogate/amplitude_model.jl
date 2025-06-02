using Flux
using DiffResults
using ForwardDiff
using LinearAlgebra

export SimpleMLPAmpModel

DEVICE = cpu

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

struct SimpleMLPAmpModel <: AbstractAmplitudeSurrogateModel
    embedding::Chain
    head::Chain
    fourier_embedding_lo::AbstractMatrix
    fourier_embedding_mid::AbstractMatrix
    fourier_embedding_hi::AbstractMatrix
end

Flux.@layer SimpleMLPAmpModel trainable=(embedding, head) #trainable=(model, fourier_embedding)

function SimpleMLPAmpModel(;n_features=6, n_hidden=1024, n_layers=2, non_lin="relu", dropout=0.0, use_skip=false, fourier_embedding_dim=128)
    non_lin_lookup = Dict("relu" => NNlib.relu, "tanh" => NNlib.tanh, "sigmoid" => NNlib.sigmoid, "celu" => NNlib.celu, "gelu" => NNlib.gelu, "swish" => NNlib.swish, "selu" => NNlib.selu)
    non_lin = non_lin_lookup[non_lin]

    function _make_hidden_block(use_skip)
        if use_skip
            hidden_block = [
                #BatchNorm(n_hidden),
                SkipConnection(Dense(n_hidden, n_hidden, non_lin), +),
                Dropout(dropout)]
        else
            hidden_block = [
                #BatchNorm(n_hidden),
                Dense(n_hidden, n_hidden, non_lin),
                Dropout(dropout)]
        end
    end

    hidden_blocks = reduce(vcat, [_make_hidden_block(use_skip) for _ in 1:n_layers])

    embedding = Chain(
        Dense(2*fourier_embedding_dim, n_hidden, non_lin),
        Dropout(dropout),
        hidden_blocks...,
    )

    head = Chain(Dense(n_hidden*3, 16),)

    embedding_matrix_lo = randn(Float32, fourier_embedding_dim, n_features) * 0.1
    embedding_matrix_mid = randn(Float32, fourier_embedding_dim, n_features) * 1
    embedding_matrix_hi = randn(Float32, fourier_embedding_dim, n_features) * 20
    #embedding_matrix *= scale

    return SimpleMLPAmpModel(embedding, head, embedding_matrix_lo, embedding_matrix_mid, embedding_matrix_hi)
end

#=
function SimpleMLPAmpModel(hparams::SimpleMLPAmpModelParams)
    return SimpleMLPAmpModel(n_features=hparams.n_features, n_hidden=hparams.n_hidden, non_lin=hparams.non_linearity)
end
=#

# Define a loss function (example: mean squared error)
function loss(model::SimpleMLPAmpModel, batch)

    preds = model(batch.x)
    targets = batch.nhits

    log_poisson_loss = -10 .^preds + 10 .^targets * preds
    return mean(log_poisson_loss)

    #return Flux.mse(preds, targets)

end



function (m::SimpleMLPAmpModel)(x)
    #return m.model(fourier_input_mapping(x, m.fourier_embedding))


    embedding_lo = m.embedding(fourier_input_mapping(x, m.fourier_embedding_lo))
    embedding_mid = m.embedding(fourier_input_mapping(x, m.fourier_embedding_mid))
    embedding_hi = m.embedding(fourier_input_mapping(x, m.fourier_embedding_hi))

    embedding = vcat(embedding_lo, embedding_mid, embedding_hi)

    return m.head(embedding)

    #angles_embedded = fourier_input_mapping(x[3:6, :], m.fourier_embedding)

    #new_x = vcat(x[1:2, :], angles_embedded)

    #return m.model(new_x)
end

function (m::SimpleMLPAmpModel)(target_position, particle_pos, particle_dir, particle_energy)
    input = collect(create_model_input(m, particle_pos, particle_dir, particle_energy, target_position))
    return cpu(m(DEVICE(input)))
end

function check_normalized(vars...)
    for var in vars
        if (var < 0) | (var > 1)
            error("Variable outside model range")
        end
    end
end


function create_model_input!(
    model::SimpleMLPAmpModel,
    particles::AbstractVector{<:Particle},
    targets::AbstractVector,
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
    return output
end


function create_model_input(::SimpleMLPAmpModel, particle_pos::AbstractVector, particle_dir::AbstractVector, particle_energy::Real, target_pos::AbstractVector)
    rel_pos = particle_pos - target_pos
    dist = norm(rel_pos)

    pos_theta, pos_phi = cart_to_sph(rel_pos ./ dist)
    dir_theta, dir_phi = cart_to_sph(particle_dir)

    # normalize

    ndist = log10(dist) / log10(300)
    ne = log10(particle_energy) / 7
    #npt = (cos(pos_theta) + 1) / 2
    #npp = pos_phi / (2π)
    #ndt = (cos(dir_theta) +1) / 2
    #ndp = dir_phi / (2π)
    #check_normalized(ndist, ne, npt, npp, ndt, ndp)

    #check_normalized(ndist, ne)
    

       #return f32(ndist), f32(ne), f32(npt), f32(npp), f32(ndt), f32(ndp)
    return ndist, ne, pos_theta / (π), pos_phi/ (2π), dir_theta / pi, dir_phi / (2π)
end

function create_model_input(model::SimpleMLPAmpModel, particle::Particle, target::PhotonTarget) 
    return create_model_input(model, particle.position, particle.direction, particle.energy, target.shape.position)

end


function get_log_amplitudes(model::SimpleMLPAmpModel, particle_pos, particle_dir, particle_energy, target_pos)
    return model(target_pos, particle_pos, particle_dir, particle_energy)
end


function get_log_amplitudes!(model::SimpleMLPAmpModel, particles, targets, feat_buffer)

    n_pmt = get_pmt_count(eltype(targets))

    #TODO: get rid of this
    input_size = 6 

    amp_buffer = create_model_input!(model, particles, targets, feat_buffer)

    input = amp_buffer
    input = permutedims(input)'

    log_expec_per_src_trg::Matrix{eltype(input)} = model(input)

    log_expec_per_src_pmt_rs = reshape(
        log_expec_per_src_trg,
        n_pmt, length(particles), length(targets))

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_pmt_rs, dims=2)

    return log_expec_per_pmt, log_expec_per_src_pmt_rs
end

function get_log_amplitudes(model::SimpleMLPAmpModel, particles, targets)
    feat_buffer = zeros(6, length(particles) *length(targets))
    return get_log_amplitudes!(model, particles, targets, feat_buffer)
end



function calculate_fisher_matrix!(model::SimpleMLPAmpModel, fisher_matrix, mod_pos, pmt_directions, particle_pos, particle_dir, log10_particle_energy, diff_res)
    pos_x, pos_y, pos_z = particle_pos
    dir_theta, dir_phi = cart_to_sph(particle_dir)

   
    function eval_model(x)
        @views input = create_model_input(model, x[1:3], sph_to_cart(x[4], x[5]), 10^x[6], mod_pos)
        return 10 .^model(collect(input))
    end

    jac_conf = ForwardDiff.JacobianConfig(eval_model, [pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy], ForwardDiff.Chunk(6))
      
    ForwardDiff.jacobian!(
        diff_res,
        eval_model,
        [pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy],
        jac_conf)

    jac = DiffResults.jacobian(diff_res)
    model_eval = DiffResults.value(diff_res)
       
    grad_mat = reshape(jac, 16, 6, 1) .* reshape(jac, 16, 1, 6)

    fisher_matrix .+= dropdims(sum(1 ./ model_eval .* grad_mat, dims=1), dims=1)
    return fisher_matrix, sum(model_eval)
end

create_diff_result(::SimpleMLPAmpModel, T::Type) = DiffResults.JacobianResult(zeros(T, 16), zeros(T, 6))
using LinearAlgebra
using NeutrinoTelescopeBase
using NaNMath
using PreallocationTools

export AnalyticAmplitudeSurrogate


struct AnalyticAmplitudeSurrogate{T, U} <: AbstractPerPMTAmplitudeSurrogateModel
    pmt_positions::Vector{T}
    rotation_matrices::Vector{Matrix{U}}
end

function AnalyticAmplitudeSurrogate(pmt_positions::AbstractVector)
    T = eltype(first(pmt_positions))
    rot_matrices = Matrix{T}[]

    for pmt_pos in pmt_positions
        R = calc_rot_matrix(pmt_pos, [0, 0, 1])
        push!(rot_matrices, R)
    end
    return AnalyticAmplitudeSurrogate(pmt_positions, rot_matrices)
end


function _sr_amplitude(dist, energy, pos_rot_ct, pos_rot_phi, dir_rot_ct, dir_rot_phi) 
    (/)(energy, (*)(dist, (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 4.128345631034468)))
end


function _sr_amplitude_grad(dist, energy, pos_rot_ct, pos_rot_phi, dir_rot_ct, dir_rot_phi) 
    (SymbolicUtils.Code.create_array)(Array, nothing, Val{1}(), Val{(6,)}(), (/)(1, (*)(dist, (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 4.128345631034468))), (*)((*)(-1, (+)((/)((*)((*)((*)(8.256691262068935, dist), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 3.1283456310344677)), (-)(dist, -71.7955290982878)), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 4.128345631034468))), (/)(energy, (*)((^)(dist, 2), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 8.256691262068935)))), (*)((*)((*)((*)((*)((*)((*)(-8.256691262068935, dist), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 3.1283456310344677)), (-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2))), (/)((^)((-)(dist, -71.7955290982878), 2), (*)((^)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), 2), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 4)))), (/)(energy, (*)((^)(dist, 2), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 8.256691262068935)))), (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct))), (-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct))))), (*)((*)((*)((*)((*)((*)((*)(138.50652627859583, dist), pos_rot_ct), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 3.1283456310344677)), (-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2))), (/)((^)((-)(dist, -71.7955290982878), 2), (*)((^)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), 2), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 4)))), (/)(energy, (*)((^)(dist, 2), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 8.256691262068935)))), (-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct))))), (*)((*)((*)((*)((*)((*)((*)(-4.118461820931911, dist), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 3.1283456310344677)), NaNMath.sin((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi)))), (/)((^)((-)(dist, -71.7955290982878), 2), (*)((^)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), 2), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 4)))), (/)(energy, (*)((^)(dist, 2), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 8.256691262068935)))), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2)), NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi)))), (*)((*)((*)((*)((*)((*)((*)(4.118461820931911, dist), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 3.1283456310344677)), NaNMath.sin((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi)))), (/)((^)((-)(dist, -71.7955290982878), 2), (*)((^)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), 2), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 4)))), (/)(energy, (*)((^)(dist, 2), (NaNMath.pow)((/)((^)((-)(dist, -71.7955290982878), 2), (*)((-)(1.8590829266769209, (^)(NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi))), 2)), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2))), 8.256691262068935)))), (^)((-)(-45.348292754758916, (*)(pos_rot_ct, (-)(15.277448579698412, (*)(16.7750642336467, dir_rot_ct)))), 2)), NaNMath.cos((*)(-0.4988029333071999, (-)(pos_rot_phi, dir_rot_phi)))))
end

function _sr_amplitude(x::AbstractVector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}

    vs = ForwardDiff.value.(x)

    ps = stack(ForwardDiff.partials, x; dims=1)
    val = _sr_amplitude(vs...)

    jvp = _sr_amplitude_grad(vs...)' * ps

    ret_val = map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) 
    end

    return first(ret_val)
end



function (m::AnalyticAmplitudeSurrogate)(dist, energy, pos_rot_ct, pos_rot_phi, dir_rot_ct, dir_rot_phi)
    return _sr_amplitude(dist, energy, pos_rot_ct, pos_rot_phi, dir_rot_ct, dir_rot_phi)
end

(m::AnalyticAmplitudeSurrogate)(x::AbstractVector) = m(x[1], x[2], x[3], x[4], x[5], x[6])
(m::AnalyticAmplitudeSurrogate)(x::AbstractMatrix) = m.(eachrow(x))

function create_model_input!(
    model::AnalyticAmplitudeSurrogate,
    position::AbstractArray{<:Real},
    direction::AbstractArray{<:Real},
    energy::Real,
    target_position::AbstractArray{<:Real},
    input_buffer) 
    
    rel_pos = position - target_position
    dist = norm(rel_pos)
    for (R, eval_slice) in zip(model.rotation_matrices, eachrow(input_buffer))

        pos_rot = R * rel_pos
        dir_rot = R * direction

        pos_rot ./= norm(pos_rot)

        pos_rot_th, pos_rot_ph = cart_to_sph(pos_rot)
        dir_rot_th, dir_rot_phi = cart_to_sph(dir_rot)

        eval_slice .= (dist, energy, cos(pos_rot_th), pos_rot_ph, cos(dir_rot_th), dir_rot_phi)
    end
    return input_buffer
end

function create_model_input(
    model::AnalyticAmplitudeSurrogate,
    position::AbstractArray{<:Real},
    direction::AbstractArray{<:Real},
    energy::Real,
    target_position::AbstractArray{<:Real},
) 
    T = promote_type(eltype(position), eltype(direction), typeof(energy), eltype(target_position))
    return create_model_input!(model, position, direction, energy, target_position, zeros(T, 16, 6))
end


function create_model_input(model::AnalyticAmplitudeSurrogate, particle::Particle, target::PhotonTarget)
    return create_model_input(
        model,
        particle.position,
        particle.direction,
        particle.energy,
        target.shape.position)
end


function get_log_amplitudes(model::AnalyticAmplitudeSurrogate, particle_pos, particle_dir, particle_energy, target_pos) 
    eval_array = create_model_input(model, particle_pos, particle_dir, particle_energy, target_pos)
    eval_model = model.(eachrow(eval_array))
    return log10.(eval_model)
end

function get_log_amplitudes(model::AnalyticAmplitudeSurrogate, particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget})

    n_pmt = get_pmt_count(eltype(targets))
    log_expec_per_src_trg = zeros(n_pmt, length(particles), length(targets))

    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        eval_array = create_model_input(model, particle, target)
        log_expec_per_src_trg[:, p_ix, t_ix] .= log10.(model.(eachrow(eval_array)))
    end

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_trg, dims=2)
    return log_expec_per_pmt, log_expec_per_src_trg
end


function calculate_fisher_matrix!(model::AnalyticAmplitudeSurrogate, fisher_matrix, mod_pos, pmt_directions, particle_pos, particle_dir, log10_particle_energy, diff_res, input_buffer)
    pos_x, pos_y, pos_z = particle_pos
    #dir_x, dir_y, _ = particle_dir

    dir_theta, dir_phi = sph_to_cart(particle_dir)
    function eval_model(x)
        #dir_x = x[4]
        #dir_y = x[5]
        #dir_z = sqrt(1 - dir_x^2 - dir_y^2)
        T = promote_type(eltype(x), eltype(mod_pos))
        buf = get_tmp(input_buffer, T)
        @views input = create_model_input!(model, x[1:3], sph_to_cart(x[4], x[5]), 10^x[6], mod_pos, buf)
        return model(input)
    end

    jac_conf = ForwardDiff.JacobianConfig(eval_model, [pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy], ForwardDiff.Chunk(6))
      
    ForwardDiff.jacobian!(
        diff_res,
        eval_model,
        [pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy],
        jac_conf)

    jac = DiffResults.jacobian(diff_res)
    model_eval = DiffResults.value(diff_res)
        
    #grad_mat = reshape(jac, 16, 6, 1) .* reshape(jac, 16, 1, 6)
    #add_mat = dropdims(sum(1 ./ model_eval .* grad_mat, dims=1), dims=1)

    #fisher_matrix .+= add_mat

    #add_mat_2 = zeros(6, 6)
    for i in 1:6
        for j in 1:6
            for k in 1:16
                fisher_matrix[i, j] += (jac[k, i] .* jac[k, j]) / model_eval[k]
            end
        end
    end

    return fisher_matrix, sum(model_eval)
end


function create_diff_result(::AnalyticAmplitudeSurrogate, T::Type)
    return DiffResults.JacobianResult(zeros(T, 16), zeros(T, 6))
end

function create_input_buffer(::AnalyticAmplitudeSurrogate, chunk_size=2)
    buffer = zeros(Float64, 16, 6)
    return DiffCache(buffer, chunk_size)
end
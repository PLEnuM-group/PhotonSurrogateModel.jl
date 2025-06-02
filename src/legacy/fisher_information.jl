using ForwardDiff
using DiffResults
using PhysicsTools
export calculate_fisher_matrix!


"""
    calculate_fisher_matrix!(m::SRSurrogateModel, fisher_matrix, mod_pos, pmt_positions, particle_pos, particle_dir, log10_particle_energy; diff_res)

Calculate the Fisher information matrix for a given surrogate model.

# Arguments
- `m::SRSurrogateModel`: The surrogate model used for evaluation.
- `fisher_matrix`: The Fisher information matrix to be updated.
- `mod_pos`: The module position.
- `pmt_directions`: An array of positions of the PMTs.
- `particle_pos`: The position of the particle.
- `particle_dir`: The direction of the particle.
- `log10_particle_energy`: The logarithm (base 10) of the particle's energy.
- `diff_res`: A pre-allocated result for storing the gradient computation.

# Returns
- `fisher_matrix`: The updated Fisher information matrix.

# Description
This function computes the Fisher information matrix assuming a poissonian likelihood by evaluating the gradient of the surrogate model with respect to the particle's position, direction, and energy.
The gradient is computed using ForwardDiff, and the Fisher matrix is updated iteratively for each PMT position.
"""
function calculate_fisher_matrix!(m::SRSurrogateModel, fisher_matrix, mod_pos, pmt_directions, particle_pos, particle_dir, log10_particle_energy, diff_res)
    pos_x, pos_y, pos_z = particle_pos
    dir_theta, dir_phi = cart_to_sph(particle_dir)

    @views eval_f(x) = m(mod_pos, x[1:3], x[4:6], sph_to_cart(x[7], x[8]), 10 ^x[9])

    ppos = pmt_directions[1]
    grd_conf = ForwardDiff.GradientConfig(eval_f, [ppos[1], ppos[2], ppos[3], pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy], ForwardDiff.Chunk(9))

    total_pred = zero(eltype(fisher_matrix))
    for ppos in pmt_directions
       
        ForwardDiff.gradient!(
            diff_res,
            eval_f,
            [ppos[1], ppos[2], ppos[3], pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy],
            grd_conf)

        @views grd = DiffResults.gradient(diff_res)[4:end]
        pred = DiffResults.value(diff_res)

        pred = clamp(pred, 1E-9)
        

        total_pred += pred

        one_o_pred = 1/pred
        @inbounds @fastmath for i in 1:6
            for j in 1:6
                fisher_matrix[i, j] += one_o_pred * grd[i] * grd[j]
            end
        end
    end

    return fisher_matrix, total_pred
end




function calculate_fisher_matrix!(m::PoissonExpFourierModel, fisher_matrix, mod_pos, pmt_directions, particle_pos, particle_dir, log10_particle_energy, diff_res)

    pos_x, pos_y, pos_z = particle_pos
    dir_theta, dir_phi = cart_to_sph(particle_dir)


    function eval_f(x)

        particle_pos = x[1:3]

        dir_theta = x[4]
        dir_phi = x[5]
        particle_dir = sph_to_cart(dir_theta, dir_phi)

        log10_particle_energy = x[6]

        rel_pos = particle_pos .- mod_pos
        dist = norm(rel_pos)
        normed_rel_pos = rel_pos ./ dist


        return m([log(dist), log10_particle_energy, particle_dir[1], particle_dir[2], particle_dir[3], normed_rel_pos[1], normed_rel_pos[2], normed_rel_pos[3], 1f0, 1f0])

    end


    grd_conf = ForwardDiff.GradientConfig(eval_f, [pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy], ForwardDiff.Chunk(6))

    total_pred = zero(eltype(fisher_matrix))

    
    ForwardDiff.gradient!(
        diff_res,
        eval_f,
        [pos_x, pos_y, pos_z, dir_theta, dir_phi, log10_particle_energy],
        grd_conf)

        grd = DiffResults.gradient(diff_res)
        pred = DiffResults.value(diff_res)
        
        total_pred += pred

        one_o_pred = 1/pred
        @inbounds @fastmath for i in 1:6
            for j in 1:6
                fisher_matrix[i, j] += one_o_pred * grd[i] * grd[j]
            end
        end

    return fisher_matrix, total_pred
end

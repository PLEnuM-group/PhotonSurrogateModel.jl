using PhotonPropagation
using PhotonSurrogateModel
using NeutrinoTelescopeBase
using NeutrinoSurrogateModelData
using PhysicsTools
using StaticArrays
using LinearAlgebra
using Random
using Flux
using DataFrames
using CairoMakie
using BSON
using LogExpFunctions
using Distributions
using DiffResults
using ForwardDiff
using Optimisers

amp_model = BSON.load("/home/wecapstor3/capn/capn100h/simple_mlp_amp_model.bson")[:model]
feat_buffer = zeros(Float32, 6, 100)

dir_ct = 0.2
dir_phis = 0:0.01:2π
target = POM(SA[0., 0., 0.], 1);

amps = []

for dir_phi in dir_phis
    pdir = sph_to_cart(acos(dir_ct), dir_phi)
    particle = Particle(SA[0., 10, 0], pdir, 0., 4E3, 0., PEMinus)
    log_amps = get_log_amplitudes([particle], [target], amp_model, feat_buffer=feat_buffer, device=cpu)
    push!(amps, exp.(log_amps[1]))
end
    
fig = Figure(size=(1000, 1000))
axes = []
for pmt_ix in 1:16
    row, col = divrem(pmt_ix-1, 4)
    ax = Axis(fig[row, col], xlabel="phi", ylabel="amp")
    push!(axes, ax)
end


for pmt_ix in 1:16
    this_amps = [a[pmt_ix] for a in amps]
    lines!(axes[pmt_ix], dir_phis, this_amps)
end
fig


fm = zeros(6, 6)

dir_phi = 3.2
pdir = sph_to_cart(acos(dir_ct), dir_phi)
particle = Particle(SA[0., 10, 0], pdir, 0., 4E3, 0., PEMinus)


diff_res = DiffResults.JacobianResult(zeros(16), zeros(6))

calculate_fisher_matrix!(amp_model, fm, target.shape.position, nothing, particle.position, particle.direction, log10(particle.energy), diff_res)

sqrt.(diag(inv(fm)))[5]

phi_res = []

for dir_phi in dir_phis
    pdir = sph_to_cart(acos(dir_ct), dir_phi)
    particle = Particle(SA[0., 10, 0], pdir, 0., 4E3, 0., PEMinus)
    fm = zeros(6, 6)
    calculate_fisher_matrix!(amp_model, fm, target.shape.position, nothing, particle.position, particle.direction, log10(particle.energy), diff_res)
    push!(phi_res, sqrt.(diag(inv(fm)))[5])
end

plot(dir_phis, phi_res)

particle = Particle(SA[0., 7, 0], pdir, 0., 4E3, 0., PEMinus)

function eval_fisher(xyz_positions)

    in_size = size(xyz_positions, 1)

    xyz_positions = reshape(xyz_positions, Int64(in_size / 3), 3)

    diff_res = DiffResults.JacobianResult(zeros(eltype(xyz_positions), 16), zeros(eltype(xyz_positions), 6))
    target_positions = xyz_positions
    total_fisher = zeros(eltype(xyz_positions), 6, 6)
    fm = zeros(eltype(xyz_positions), 6, 6)
    for tpos in eachrow(target_positions)
        avg_fisher = zeros(eltype(xyz_positions), 6, 6)
        for it in 1:10
            ppos = rand(Uniform(-10, 10), 3)
            particle = Particle(ppos, pdir, 0., 4E3, 0., PEMinus)        
            calculate_fisher_matrix!(amp_model, fm, tpos, nothing, particle.position, particle.direction, log10(particle.energy), diff_res)
        
            if all(isfinite.(fm))
                avg_fisher += fm
            end
        end

        if all(isfinite.(avg_fisher))
            total_fisher += avg_fisher / 10
        end

    end

    if isposdef(total_fisher)
        return -tr(inv(total_fisher))
    end

    return 1E9
end

model = (;positions = 40 * rand(10, 3) .- 20)
state = Optimisers.setup(Optimisers.Adam(1), model)


positions_log = []
loss_log = []
nsteps = 20
for step in 1:nsteps
    flat, re = Optimisers.destructure(model)
    grad = re(ForwardDiff.gradient(eval_fisher, flat))
    state, model = Optimisers.update(state, model, grad)
    push!(positions_log, model.positions)
end


fig = Figure()
ax = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = fig[2, :]

for (step, positions) in enumerate(positions_log)
    alpha = 0.1 + 0.9*(step / nsteps)
    for pos in eachrow(positions)
        scatter!(ax, pos[1], pos[2], color=(:green, alpha))
        scatter!(ax2, pos[1], pos[3], color=(:green, alpha))
    end
end
#scatter!(ax, 0, 7, color=:black, markersize=20)
#scatter!(ax2, 0, 7, color=:black, markersize=20)
fig





    
    scatter!(ax, p0[2, 1], p0[2, 2], color=(:blue, alpha))
    scatter!(ax, p0[3, 1], p0[3, 2], color=(:green, alpha))
    scatter!(ax, p0[4, 1], p0[4, 2], color=(:magenta, alpha))
using SymbolicRegression
using SymbolicUtils
using CairoMakie
using BSON
using PhotonSurrogateModel
using PhotonPropagation
using Distributions
using JLD2
using Flux
using PhysicsTools
using StaticArrays
using Rotations
using LinearAlgebra
using NaNMath
using LoopVectorization
using Bumper
using ForwardDiff
using DiffResults
using NeutrinoTelescopeBase
using Symbolics
import SymbolicRegression: compute_complexity

include("sr_utils.jl")

model = BSON.load("/home/wecapstor3/capn/capn100h/simple_mlp_amp_model.bson")[:model]

device = cpu
target = POM(SA[0., 0., 0.], 1)
pmt_pos = get_pmt_positions(target, RotMatrix3(I))


res = load("/home/wecapstor3/capn/capn100h/sr_out.jld2")
hof = res["hof"]
opt = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=[exp, NaNMath.acos, tan, NaNMath.atan, sqrt, square, cos, tanh, expsq, expabs],
    nested_constraints = [square => [square => 1], sqrt => [sqrt => 1]],
    populations=3*1,
    population_size=150,
    ncycles_per_iteration=200,
    turbo = true,
    mutation_weights = MutationWeights(optimize=0.005),
    bumper = true,
    complexity_of_constants=1,
    complexity_of_variables=1,
    complexity_of_operators = [
        (^) => 2,
    ],
    parsimony = 0.01,
    adaptive_parsimony_scaling=1000,
    elementwise_loss=LogL2Loss(),
    save_to_file = true,
    progress=true,
    #fraction_replaced_hof=0.15,
    dimensional_constraint_penalty=1000,
    use_frequency=true,
    #batching=true,
    #batch_size=1000,
    maxsize=50,
    #warmup_maxsize_by=0.1
)
dominating = calculate_pareto_frontier(hof)


function make_scan_inputs(params)
   
    sr_eval = []
    sur_eval = Particle[]

    for param in params
        dir = sph_to_cart(acos(param.dir_ct), param.dir_phi)
        pos = param.dist .* sph_to_cart(acos(param.pos_ct), param.pos_phi)

        
        pmt_p = pmt_pos[param.pmt_ix]
        
        R = calc_rot_matrix(pmt_p, [0, 0, 1])

        pos_rot = R * pos
        dir_rot = R * dir

        pos_rot_sph = cart_to_sph(pos_rot ./ norm(pos_rot))
        dir_rot_sph = cart_to_sph(dir_rot)
    
        
        push!(sr_eval, [param.dist, param.energy, cos(pos_rot_sph[1]), pos_rot_sph[2], cos(dir_rot_sph[1]), dir_rot_sph[2]])
        push!(sur_eval, Particle(pos, dir, 0., energy, 0., PEPlus))
    end

    sr_eval = reduce(hcat, sr_eval)
    #sur_eval = reduce(hcat, sur_eval)

    return sr_eval, sur_eval

end


ix_sel = 50

pom_coords = Vector(sph_to_cart.(eachcol(make_pom_pmt_coordinates(Float64))))

ana_mod = PhotonSurrogateModel.AnalyticAmplitudeSurrogate(pom_coords)

# scan pos_theta
#distances = 0.5:0.5:150
dist = 35.
pos_ct = -1:0.01:1
pos_phi = 2.5
dir_ct = -0.5
dir_phi = 1.7
energy = 135000
pmt_ix = 12
params = [
    (dist=dist, pos_ct=pct, pos_phi=pos_phi, energy=energy, dir_ct=dir_ct, dir_phi=dir_phi, pmt_ix=pmt_ix) for pct in pos_ct
    ]

sr_eval, sur_eval = make_scan_inputs(params)
y_eval, _ = eval_tree_array(dominating[ix_sel].tree, sr_eval, opt)
model_eval, model_eval_rs = get_log_amplitudes(model, sur_eval, [target])
ana_model_eval, ana_model_eval_rs = get_log_amplitudes(ana_mod, sur_eval, [target])



fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
lines!(ax, pos_ct, 10 .^model_eval_rs[pmt_ix, :, 1])
lines!(ax, pos_ct, 10 .^ana_model_eval_rs[pmt_ix, :, 1])
lines!(ax, pos_ct, y_eval)
fig



# scan dir_phi
#distances = 0.5:0.5:150
dist = 25.
pos_ct = 0.3
pos_phi = 1.5
dir_ct = -.7
dir_phi = 0:0.01:2π
energy = 135000
pmt_ix = 12
params = [
    (dist=dist, pos_ct=pos_ct, pos_phi=pos_phi, energy=energy, dir_ct=dir_ct, dir_phi=dp, pmt_ix=pmt_ix) for dp in dir_phi
    ]

sr_eval, sur_eval = make_scan_inputs(params)
y_eval, _ = eval_tree_array(dominating[ix_sel].tree, sr_eval, opt)
model_eval, model_eval_rs = get_log_amplitudes(model, sur_eval, [target])
ana_model_eval, ana_model_eval_rs = get_log_amplitudes(ana_mod, sur_eval, [target])


fig = Figure()
ax = Axis(fig[1, 1],)
lines!(ax, dir_phi, 10 .^model_eval_rs[pmt_ix, :, 1])
lines!(ax, dir_phi, 10 .^ana_model_eval_rs[pmt_ix, :, 1])
lines!(ax, dir_phi, y_eval)
fig

eq_sel = dominating[30]
fig, ax, _ = lines([mem.loss for mem in dominating], axis=(;yscale=log10))
vlines!(ax, [compute_complexity(eq_sel, opt)])
fig


eq_sym = node_to_symbolic(eq_sel.tree, opt, variable_names=["dist", "energy", "pos_rot_ct", "pos_rot_phi", "dir_rot_ct", "dir_rot_phi"])
vars = Symbolics.get_variables(eq_sym)

eq_rw = Rewriters.Postwalk(literaltoreal)(eq_sym)
erw = Rewriters.Postwalk(literaltoreal).(vars)


expr = build_function(eq_rw, erw...)
expr_grad = build_function(Symbolics.gradient(eq_rw, erw), erw...)

buf = open("test.jl", "w")
Base.show_unquoted(buf, expr)
Base.show_unquoted(buf, expr_grad)
close(buf)

diff_res = create_diff_result(ana_mod, Float64)
targets = [POM(SA[0., 0., 0.], 1), POM(SA[0., 10., 0.], 2)]

phis = 0.1:0.1:2π
ppos = [1., 10., 15.]

fis_mlp = []
for phi in phis
    fm = zeros(6, 6)
    for target in targets
        calculate_fisher_matrix!(model, fm, target.shape.position, get_pmt_positions(target), ppos, sph_to_cart(0.3, phi), 3.5, diff_res)[1]
    end
    push!(fis_mlp, fm)
end

fis_ana = []
for phi in phis
    fm = zeros(6, 6)
    for target in targets
        calculate_fisher_matrix!(ana_mod, fm, target.shape.position, get_pmt_positions(target), ppos, sph_to_cart(0.3, phi), 3.5, diff_res)[1]
    end
    push!(fis_ana, fm)
end

cr_mlp = inv.(fis_mlp)

cr_ana = inv.(fis_ana)



fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
lines!(ax, phis, [diag(cr)[6] for cr in cr_ana])
lines!(ax, phis, [diag(cr)[6] for cr in cr_mlp])
fig

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
lines!(ax, phis, fis_mlp)
lines!(ax, phis, fis_ana)
fig

d


using StatsBase
using PhysicsTools
using CairoMakie
using Distributions
using Flux
using BSON
using PhotonPropagation
using LinearAlgebra
using Rotations
using StaticArrays
using PhotonSurrogateModel
using JLD2
using NeutrinoTelescopeBase
using Distributed
using TensorBoardLogger

n_workers = 16 #Threads.nthreads()
procs = addprocs(n_workers)

@everywhere include("sr_utils.jl")
@everywhere using SymbolicRegression
@everywhere using LoopVectorization
@everywhere using NaNMath
@everywhere using Bumper
@everywhere using TensorBoardLogger

logger = SRLogger(TBLogger("/home/wecapstor3/capn/capn100h/tensorboard/sr_runs/"))

model = BSON.load("/home/wecapstor3/capn/capn100h/simple_mlp_amp_model.bson")[:model]


device = cpu
target = POM(SA[0., 0., 0.], 1)
pmt_pos = get_pmt_positions(target, RotMatrix3(I))

sr_X = []
sr_y = Float64[]

for i in 1:5000
    pos_ct = rand(Uniform(-1, 1))
    pos_phi = rand(Uniform(0, 2π))
    dist = rand(Uniform(0, 200))
    #dist = 10.
    energy = 10 .^ rand(Uniform(2, 6))

    dir_ct = rand(Uniform(-1, 1))
    dir_phi = rand(Uniform(0, 2π))

    pos = dist .* sph_to_cart(acos(pos_ct), pos_phi)
    dir = sph_to_cart(acos(dir_ct), dir_phi)


    model_input = collect(create_model_input(model, pos, dir, energy, [0, 0, 0]))
    model_eval = cpu(model(model_input |> device))

    pmt_ix = rand(1:16)
    pmt_p = pmt_pos[pmt_ix]

    #inp_transformed = transform_input(pmt_p, pos, dir)

    R = calc_rot_matrix(pmt_p, [0, 0, 1])

    pos_rot = R * pos
    dir_rot = R * dir

    pos_rot_sph = cart_to_sph(pos_rot ./ norm(pos_rot))
    dir_rot_sph = cart_to_sph(dir_rot)


    #push!(sr_X, [energy, dist, inp_transformed...])
    push!(sr_X, [dist, energy, cos(pos_rot_sph[1]), pos_rot_sph[2], cos(dir_rot_sph[1]), dir_rot_sph[2]])
    
    push!(sr_y, 10 .^model_eval[pmt_ix])
end

sr_X = reduce(hcat, sr_X)

#scatter(sr_X[2, :], sr_y, color = log10.(sr_X[1, :]), axis=(;yscale=log10, xscale=log10))

#scatter(sr_X[4, :], sr_X[6, :], color = log10.(sr_y ./ sr_X[2, :]))

outdir = "/home/wecapstor3/capn/capn100h/sr_out_test"


opt = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=[exp, NaNMath.acos, tan, NaNMath.atan, sqrt, square, cos, tanh, expnegsq, expabs, log],
    nested_constraints = [square => [square => 1], sqrt => [sqrt => 1]],
    populations=3*n_workers,
    population_size=80,
    ncycles_per_iteration=250,
    turbo = true,
    mutation_weights = MutationWeights(optimize=0.003),
    bumper = true,
    complexity_of_constants=1,
    complexity_of_variables=1,
    complexity_of_operators = [
        (^) => 2,
    ],
    parsimony = 0.01,
    #adaptive_parsimony_scaling=1000,
    elementwise_loss=LogL2Loss(),
    output_directory = outdir,
    save_to_file = true,
    progress=true,
    #fraction_replaced_hof=0.15,
    dimensional_constraint_penalty=1000,
    use_frequency=true,
    #batching=true,
    #batch_size=1000,
    maxsize=60,
    #warmup_maxsize_by=0.2,
    batching=true,
    batch_size=1000,
)


state = load("/home/wecapstor3/capn/capn100h/sr_out.jld2", "state")
hof = load("/home/wecapstor3/capn/capn100h/sr_out.jld2", "hof")
saved_state = (state, hof)

#saved_state = nothing

state, hof = equation_search(
    sr_X,
    sr_y,
    niterations=10000,
    options=opt,
    parallelism=:multiprocessing,
    variable_names=["dist", "energy", "pos_rot_ct", "pos_rot_phi", "dir_rot_ct", "dir_rot_phi"],
    X_units=["m", "Constants.GeV","", "", "", ""],
    runtests=false,
    return_state=true,
    #numprocs = n_workers
    procs = procs,
    saved_state=saved_state,
    logger=logger
)
jldsave("/home/wecapstor3/capn/capn100h/sr_out.jld2", hof=hof, state=state)


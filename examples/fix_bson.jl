using Flux
using PhotonSurrogateModel
using BSON: @load, parse, raise_recursive, bson
using Glob
using NeutrinoTelescopes
function fix_amp_model(fname)

    data = parse(fname)

    model = data[:model]
    new_name = ["PhotonSurrogateModel", "PoissonExpModel"]
    model[:type]
    model[:type][:name] = new_name

    new_name = ["PhotonSurrogateModel", "Normalizer"]
    model[:data][2][:type][:name] = new_name

    tf_vec = data[:tf_vec]
    new_name = ["PhotonSurrogateModel", "Normalizer"]

    for val in values(tf_vec)
        val[:type][:name] = new_name
    end

    hparams = data[:hparams]
    new_name = ["PhotonSurrogateModel", "AbsScaPoissonExpModelParams"]
    hparams[:type][:name] = new_name
    fixed_data = raise_recursive(data, Main)
    rm(fname)
    bson(fname, fixed_data)    
end

function fix_time_model(fname)

    data = parse(fname)

    model = data[:model]
    new_name = ["PhotonSurrogateModel", "NNRQNormFlow"]
    model[:type]
    model[:type][:name] = new_name

    new_name = ["PhotonSurrogateModel", "Normalizer"]
    
    model[:data][5][:type][:name] = new_name

    tf_vec = data[:tf_vec]
    new_name = ["PhotonSurrogateModel", "Normalizer"]

    for val in values(tf_vec)
        val[:type][:name] = new_name
    end

    hparams = data[:hparams]
    new_name = ["PhotonSurrogateModel", "AbsScaRQNormFlowHParams"]
    hparams[:type][:name] = new_name
    fixed_data = raise_recursive(data, Main)
    rm(fname)
    bson(fname, fixed_data)    
end


model_dir = "/home/hpc/capn/capn100h/.julia/dev/NeutrinoSurrogateModelData/model_data/time_surrogate_perturb/lightsabre"
fix_amp_model(joinpath(model_dir, "amp.bson"))


time_models = glob("time_*", model_dir)

for time_model in time_models
    fix_time_model(time_model)
end


function fix_fisher_per_line(fname)
    model_file = fname
    data = parse(model_file)

    tf_in = data[:tf_in]
    tf_in[:type][:name] = ["PhotonSurrogateModel", "Normalizer"]


    tf_out = data[:tf_out]
    tf_out[:type][:name] = ["PhotonSurrogateModel", "Normalizer"]

    fixed_data = raise_recursive(data, Main)

    rm(model_file)
    bson(fname, fixed_data)
end

fix_fisher_per_line("/home/wecapstor3/capn/capn100h/snakemake/fisher_surrogates/fisher_surrogate_per_string_lightsabre.bson")

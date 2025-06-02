module PhotonSurrogateModel

using Reexport
using NeutrinoTelescopeBase
using PhysicsTools
using Base.Iterators
using LogExpFunctions
export AbstractPhotonSurrogateModel, AbstractAmplitudeSurrogateModel
export create_model_input, create_model_input!
export get_log_amplitudes
export calculate_fisher_matrix!
export create_diff_result
export fourier_input_mapping
export create_input_buffer

abstract type AbstractPhotonSurrogateModel end
abstract type AbstractAmplitudeSurrogateModel <: AbstractPhotonSurrogateModel end
abstract type AbstractPerPMTAmplitudeSurrogateModel <: AbstractAmplitudeSurrogateModel end

# Interface

_NI(m) = error("Not Implemented $m")

create_model_input(model::AbstractPhotonSurrogateModel, particle, target) = _NI("create_model_input")
create_input_buffer(::AbstractPhotonSurrogateModel) = _NI("create_input_buffer")

function create_model_input(
    model::AbstractPhotonSurrogateModel,
    particles::AbstractVector{<:Particle},
    targets::AbstractVector,
)
    output = zeros(6, length(particles)*length(targets))
    return create_model_input!(model, particles, targets, output)
end


get_log_amplitudes(model::AbstractAmplitudeSurrogateModel, particle_pos, particle_dir, particle_energy, target_pos) = _NI("get_log_amplitudes")

get_log_amplitudes(
    model::AbstractAmplitudeSurrogateModel,
    particle::Particle,
    target::PhotonTarget) = get_log_amplitudes(
        model,
        particle.position,
        particle.direction,
        particle.energy,
        target.shape.position
    )


get_log_amplitudes(model::AbstractAmplitudeSurrogateModel, particles::AbstractVector{<:Particle}, target::AbstractVector{:PhotonTarget}) = _NI("get_log_amplitudes")


calculate_fisher_matrix!(
    model::AbstractPhotonSurrogateModel,
    fisher_matrix,
    mod_pos,
    pmt_directions,
    particle_pos,
    particle_dir,
    log10_particle_energy,
    diff_res) = _NI("calculate_fisher_matrix!")


create_diff_result(::AbstractPhotonSurrogateModel, T::Type) = _NI(create_diff_result)


abstract type HyperParams end

function fourier_input_mapping(x, B=nothing)
    if isnothing(B)
        return x
    else
        x_proj = permutedims((2 * π * x)) * B'
        return hcat(sin.(x_proj), cos.(x_proj))'
    end
end



include("neural_surrogate/amplitude_model.jl")
include("analytic_surrogate.jl")


#=
LEGACY
@reexport using .RQSplineFlow
include("rq_spline_flow.jl")
include("photon_surrogate/utils.jl")
include("photon_surrogate/dataio.jl")
include("photon_surrogate/time_models.jl")
include("photon_surrogate/amplitude_models.jl")

include("analytic_surrogate/amplitude.jl")

include("fisher_information.jl")
=#
end

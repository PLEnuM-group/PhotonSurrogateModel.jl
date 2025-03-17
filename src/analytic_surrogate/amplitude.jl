using PhysicsTools
using PhotonPropagation
using DiffResults
using StaticArrays
using Rotations
export AnalyticSurrogateModel
export SRSurrogateModel
export create_diff_result
using ForwardDiff
using SymbolicUtils
using NaNMath
noNaNs(x::Real) = true
noNaNs(x::ForwardDiff.Dual) = !any(isnan, ForwardDiff.partials(x))


"""
    convert_pmt_frame(pmt_direction::AbstractVector, particle_pos::AbstractVector, particle_dir::AbstractVector)
Convert coordinates of the particle / PMT system to the PMT frame.

 # Arguments
 - `pmt_direction::AbstractVector`: The direction vector of the PMT.
 - `particle_pos::AbstractVector`: The position vector of the particle.
 - `particle_dir::AbstractVector`: The direction vector of the particle.

 # Returns
 - The zenith angle of the rotated particle position in spherical coordinates.
 - The zenith angle of the rotated particle direction in spherical coordinates.
 - `delta_phi`: The difference in azimuthal angles between the rotated particle position and direction.

The function first rotates the particle position and direction vectors to the PMT frame (PMT axis aligned with the z-axis).
It then converts these rotated vectors to spherical coordinates and computes the difference in their azimuthal angles.
"""
function convert_pmt_frame(pmt_direction::AbstractVector, particle_pos::AbstractVector, particle_dir::AbstractVector)

    part_pos_rot = rot_to_ez_fast(pmt_direction, particle_pos)
    part_dir_rot = rot_to_ez_fast(pmt_direction, particle_dir)
    
    #=
    part_pos_rot_sph = cart_to_sph(part_pos_rot ./ norm(part_pos_rot))
    part_dir_rot_sph = cart_to_sph(part_dir_rot)
    delta_phi = part_pos_rot_sph[2] - part_dir_rot_sph[2]

    return part_pos_rot_sph[1], part_dir_rot_sph[1], delta_phi
    =#
    # Convert the rotated particle position to cylindrical coordinates
    pos_cyl = cart_to_cyl(part_pos_rot)

    # Calculate Rotation matrix that rotates the particle direction to the xz-plane
    rotm = RotZ(-pos_cyl[2])

    # Apply the rotation matrix to the particle direction
    part_dir_rot_xz = rotm * part_dir_rot
    part_dir_rot_xz_sph = cart_to_sph(part_dir_rot_xz)

    # We dont have to apply the rotation matrix to the particle position as we are only interested in the zenith angle
    part_pos_rot_sph = cart_to_sph(part_pos_rot ./ norm(part_pos_rot))

    return part_pos_rot_sph[1], part_dir_rot_xz_sph[1], part_dir_rot_xz_sph[2]
end


abstract type AnalyticSurrogateModel end

struct SRSurrogateModel <: AnalyticSurrogateModel end


function _sr_amplitude(energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale)
    return sqrt(((energy + 196.99990115571856) * ((((24.071180505247334 / sqrt((abs(3.1375769542264234 - phi_dir) + (((0.2527566982481768 ^ theta_dir) / 0.1047213870655026) / (theta_pos + (0.05966138001737878 ^ ((theta_pos + 0.26296689034611676) * theta_dir))))) + tanh((((theta_pos * theta_pos) * (theta_dir * -0.16835412713452186)) ^ 2) * 0.7551665125771176))) / (distance - (((theta_dir ^ theta_pos) - -2.0003863609058623) / -0.0860817592408263))) ^ ((theta_pos * 1.552909303464834) + 6.875484398708654)) * 314.1420956231935)) ^ 1.818085535256962)
end


function _sr_amplitude_grad(energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale)
    begin
        #= /home/hpc/capn/capn100h/.julia/packages/SymbolicUtils/8d6hE/src/code.jl:468 =#
        (SymbolicUtils.Code.create_array)(Array, nothing, Val{1}(), Val{(7,)}(), (/)(0.25756487974466313, (*)((*)((^)(energy, 0.16311839922115912), (+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2)))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-0.8277027096616162, theta_dir)))), (*)((*)(-1, (+)((*)((+)((/)((*)(if (signbit)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos)))
                                            -1
                                        else
                                            1
                                        end, (+)(-0.08649012148556141, (/)((*)(0.11267100122049988, (exp)((*)(0.11267100122049988, distance))), theta_dir))), theta_pos), (/)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), (sinh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance)))), (*)((*)(2, NaNMath.sqrt(distance)), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2)))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-0.8277027096616162, theta_dir))), (*)((*)((*)((*)(-0.14914900883519233, (+)(-0.8277027096616162, theta_dir)), (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.02224542683652028, (^)(distance, 2)))), (+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2)))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-1.827702709661616, theta_dir))))), (/)((*)(0.30776740641084865, (^)(energy, 0.8368816007788409)), (*)((^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (*)(2, (+)(-0.8277027096616162, theta_dir))), (^)((+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2))), 2)))), (*)((*)(-1, (+)((/)((*)((*)((*)((+)(-0.8277027096616162, theta_dir), (+)((*)((*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos)), (^)(theta_pos, (+)(-1, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))))), (*)((*)(1.8760043466558647, (^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos)))), NaNMath.log(theta_pos)))), (+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2)))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-1.827702709661616, theta_dir))), (*)(0.14914900883519233, distance)), (*)((+)((/)((*)((*)(-0.3475526750813855, NaNMath.sqrt(distance)), (sinh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance)))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2)), (*)((*)(-1, if (signbit)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos)))
                                            -1
                                        else
                                            1
                                        end), (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), (^)(theta_pos, 2)))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-0.8277027096616162, theta_dir))))), (/)((*)(0.30776740641084865, (^)(energy, 0.8368816007788409)), (*)((^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (*)(2, (+)(-0.8277027096616162, theta_dir))), (^)((+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2))), 2)))), (*)((*)(-1, (+)((/)((*)((*)((*)(-1, if (signbit)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos)))
                                            -1
                                        else
                                            1
                                        end), (/)((exp)((*)(0.11267100122049988, distance)), (^)(theta_dir, 2))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-0.8277027096616162, theta_dir))), theta_pos), (*)((+)((/)((*)((*)((*)((*)(1.8760043466558647, (+)(-0.8277027096616162, theta_dir)), (^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos)))), NaNMath.log(theta_pos)), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-1.827702709661616, theta_dir))), (*)(0.14914900883519233, distance)), (*)(NaNMath.log((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance)))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-0.8277027096616162, theta_dir)))), (+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2)))))), (/)((*)(0.30776740641084865, (^)(energy, 0.8368816007788409)), (*)((^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (*)(2, (+)(-0.8277027096616162, theta_dir))), (^)((+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2))), 2)))), (/)((*)((*)((*)(6.178604399619335, (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 4))), (^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (+)(-0.8277027096616162, theta_dir))), (/)((*)(0.30776740641084865, (^)(energy, 0.8368816007788409)), (*)((^)((+)(0.23508448098889415, (/)((^)(theta_pos, (*)(1.8760043466558647, (+)((+)(-1.4330390227977239, theta_dir), theta_pos))), (*)(0.14914900883519233, distance))), (*)(2, (+)(-0.8277027096616162, theta_dir))), (^)((+)((abs)((+)(-0.49242697825019954, (/)((+)((*)(-0.08649012148556141, distance), (/)((exp)((*)(0.11267100122049988, distance)), theta_dir)), theta_pos))), (/)((cosh)((*)((*)(-0.3475526750813855, (+)(2.2792764022764818, theta_pos)), NaNMath.sqrt(distance))), (^)((/)(1.7576410895884482, (+)(3.148056563571116, (*)(-1, phi_dir))), 2))), 2)))), (^)((+)(3.148056563571116, (*)(-1, phi_dir)), 3)), 0, 0)
    end
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



function (m::SRSurrogateModel)(energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale)
    #return ((((energy / 64.01339842100778) ^ 0.8935288620057904) / (exp(sqrt(distance * 1.475613689574574)) - 2.2614466839386034)) * (exp((((((((theta_pos ^ theta_dir) - 1.6732972805419357) + (cos(delta_phi * 0.999931998910065) * (((cos(-2.1605025756967993 - ((theta_dir + theta_pos) * -0.8961414254923246)) - (cos(delta_phi / 0.9892881317933397) * 0.4447736144973129)) * theta_pos) + 0.22875138406989487))) * 0.6955551530471992) * theta_dir) - sqrt(theta_pos)) + ((theta_dir / distance) * 1.7763433006250346)) * -1.817915092655903) + (theta_dir ^ (theta_dir * -2.1114647904144155)))) * 1.2169981536928478

    return _sr_amplitude(energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale)
end


function (m::SRSurrogateModel)(x)
    return _sr_amplitude(x)
end

function convert_model_inputs(m::SRSurrogateModel, target_pos::AbstractVector, pmt_pos::AbstractVector, particle_pos::AbstractVector, particle_dir::AbstractVector, particle_energy::Real)
    rpos_x = particle_pos[1] - target_pos[1]
    rpos_y = particle_pos[2] - target_pos[2]
    rpos_z = particle_pos[3] - target_pos[3]

    rpos = SVector{3}(rpos_x, rpos_y, rpos_z)

    distance = hypot(rpos...)
    theta_pos, theta_dir, phi_dir = convert_pmt_frame(pmt_pos, rpos, particle_dir)

    return particle_energy, distance, theta_pos, theta_dir, phi_dir
end


function (m::SRSurrogateModel)(target_pos::AbstractVector, pmt_pos::AbstractVector, particle_pos::AbstractVector, particle_dir::AbstractVector, particle_energy::Real)
    abs_scale = 1.0
    sca_scale = 1.0

    particle_energy, distance, theta_pos, theta_dir, phi_dir = convert_model_inputs(m, target_pos, pmt_pos, particle_pos, particle_dir, particle_energy)
    model_eval = m(particle_energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale)
    #model_eval = m(SA[particle_energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale])

    return model_eval
end



function (m::AnalyticSurrogateModel)(p::Particle, target_pos::AbstractVector, pmt_pos::AbstractVector)
    return @fastmath m(target_pos, pmt_pos, p.position, p.direction, p.energy)
end

function create_diff_result(::SRSurrogateModel, T)
    return DiffResults.GradientResult(zeros(T, 9))
end
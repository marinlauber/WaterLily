module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult!

include("Flow.jl")
export Flow,mom_step!

include("Body.jl")
export AbstractBody,measure_sdf!

include("AutoBody.jl")
export AutoBody,measure,sdf,+,-

include("Metrics.jl")

# include("Coupling.jl")
# export Relaxation, IQNCoupling, update, res

abstract type AbstractSimulation end

"""
    Simulation(dims::NTuple, u_BC::NTuple, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0., ϵ=1,
               uλ::Function=(i,x)->u_BC[i],
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `u_BC`: Simulation domain velocity boundary conditions, `u_BC[i]=uᵢ, i=eachindex(dims)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `ϵ`: BDIM kernel width.
  - `uλ`: Function to generate the initial velocity field.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array` and `CuArray` run on CPU and CUDA backends, respectively.

See files in `examples` folder for examples.
"""
struct Simulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::NTuple{N}, u_BC::NTuple{N}, L::Number;
                        Δt=0.25, ν=0., U=√sum(abs2,u_BC), ϵ=1,
                        uλ::Function=(i,x)->u_BC[i],
                        body::AbstractBody=NoBody(),T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,T,f=mem)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ))
    end
end

time(sim::AbstractSimulation) = sum(sim.flow.Δt[1:end-1])
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::AbstractSimulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::Simulation,t_end;remeasure=true,verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step. 
Can be set to `false` for static geometries to speed up simulation.
"""
function sim_step!(sim::Simulation,t_end;verbose=false,remeasure=true)
    t = time(sim)
    while t < t_end*sim.L/sim.U
        remeasure && measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

"""
    measure!(sim::Simulation,t=time(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::AbstractSimulation,t=time(sim))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois)
end

export AbstractSimulation,Simulation,sim_step!,sim_time,measure!

# default WriteVTK functions
function vtkWriter end
function write! end
function default_attrib end
function pvd_collection end
# export
export vtkWriter, write!, default_attrib

# default ReadVTK functions
function restart_sim! end
# export
export restart_sim!

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/WaterLilyAMDGPUExt.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/WaterLilyCUDAExt.jl")
        @require WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192" include("../ext/WaterLilyWriteVTKExt.jl")
        @require ReadVTK = "dc215faf-f008-4882-a9f7-a79a826fadc3" include("../ext/WaterLilyReadVTKExt.jl")
    end
end

end # module

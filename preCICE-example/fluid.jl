using PreCICE
using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using WriteVTK

# structure to store fluid state
struct Store
    uˢ::AbstractArray
    pˢ::AbstractArray
    xˢ::AbstractArray
    ẋˢ::AbstractArray
    function Store(sim::AbstractSimulation)
        new(copy(sim.flow.u),copy(sim.flow.p),copy(sim.body.surf.pnts),copy(sim.body.velocity.pnts))
    end
end
function store!(s::Store,sim::AbstractSimulation)
    s.uˢ .= sim.flow.u; s.pˢ .= sim.flow.p;
    s.xˢ .= sim.body.surf.pnts; s.ẋˢ .= sim.body.velocity.pnts;
end
function revert!(s::Store,sim::AbstractSimulation)
    sim.flow.u .= s.uˢ; sim.flow.p .= s.pˢ; pop!(sim.flow.Δt)
    sim.body.surf.pnts .= s.xˢ; sim.body.velocity.pnts .= s.ẋˢ;
end

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
                            WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                            a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => _body,
    "Vorticity_Z" => vorticity,
)# this maps what to write to the name in the file

## Simulation parameters
L=2^5
Re=200
U=1
ϵ=0.5
thk=2ϵ+√2

# coupling
PreCICE.createParticipant("WaterLily", "./precice-config.xml", 0, 1)

# write for the sim
wr = vtkWriter("Inverted-Flag"; attrib=custom_attrib)

let # setting local scope for dt outside of the while loop

    # this is transposed, careful
    function force(b::DynamicBody,flow::Flow)
        reduce(hcat,[NurbsForce(b.surf,flow.p,s) for s ∈ integration_points])'
    end
    
    PreCICE.initialize()

    # get the mesh verticies from the structural solver
    (vertexIDs_nurbs, vertices_nurbs) = getMeshVertexIDsAndCoordinates("Nurbs-Mesh-Solid")
    vertexIDs_nurbs = Array{Int32}(vertexIDs_nurbs)
    (vertexIDs_forces, tmp) = getMeshVertexIDsAndCoordinates("Force-Mesh-Solid")
    integration_points = tmp[:,1] # only x-coordinate make sense
    vertexIDs_forces = Array{Int32}(vertexIDs_forces)
    (_, knots) = getMeshVertexIDsAndCoordinates("Knots-Mesh")
    knots = knots[:,1] # the other coordinates are dummies
    @show knots
    
    # construct from mesh, this can be tidy
    N,D = size(vertices_nurbs)
    u⁰ = MMatrix{D,N}(vertices_nurbs'.*L.+[2L,2L])
    nurbs = NurbsCurve(copy(u⁰),knots,ones(N))
    
    # overload the distance function and make function
    dis(p,n) = √(p'*p) - thk/2
    body = DynamicBody(nurbs, (0,1); dist=dis);

    # make a simulation and a storage
    sim = Simulation((6L,4L),(U,0),L;U,ν=U*L/Re,body,ϵ,T=Float64)
    store = Store(sim)
    writeData = force(sim.body,sim.flow)
    
    # simulations time
    t = 0.0;

    while PreCICE.isCouplingOngoing()

        # set time step
        dt_precice = PreCICE.getMaxTimeStepSize()
        dt = min(sim.flow.Δt[end], dt_precice)
        sim.flow.Δt[end] = dt

        if PreCICE.requiresWritingCheckpoint()
            store!(store,sim)
        end

        readData = PreCICE.readData("Nurbs-Mesh-Solid", "Displacements", vertexIDs_nurbs, dt)
        readData .= u⁰' + readData.*L # scale and move to correct location
        ParametricBodies.update!(body,Matrix(readData'),dt)
        
        # solver update
        measure!(sim,t); mom_step!(sim.flow,sim.pois)
        
        writeData = force(sim.body,sim.flow)
        t<=2L && writeData[:,2] .= -0.5
        PreCICE.writeData("Force-Mesh-Solid", "Forces", vertexIDs_forces, writeData)
        
        dt_precice = PreCICE.advance(dt)

        if PreCICE.requiresReadingCheckpoint()
            revert!(store,sim)
        else
            t += dt
            # write data
            write!(wr, sim)
        end    
    end # while
    close(wr)
end # let

PreCICE.finalize()
println("WaterLily: Closing Julia solver...")
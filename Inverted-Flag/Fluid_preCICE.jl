using PreCICE
using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using Plots; gr()
include("../examples/TwoD_plots.jl")
# this is transposed, careful
function force(b::DynamicBody,flow::Flow)
    reduce(hcat,[NurbsForce(b.surf,flow.p,s) for s ∈ integration_points])'
end
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

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

## Simulation parameters
L=2^5
Re=200
U=1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[2L,2L])
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1));

# make a simulation and a storage
sim = Simulation((6L,4L),(U,0),L;U,ν=U*L/Re,body,ϵ,T=Float64)
store = Store(sim)

# make a problem
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptRight, 0.0; comp=1),
    Boundary1D("Neumann", ptRight, 0.0; comp=2)
]

# make a structure
struc = DynamicFEOperator(mesh, gauss_rule, 0.35, 1000, Dirichlet_BC, Neumann_BC, ρ=(x)->5.0; ρ∞=0.5)
# 
# location of integration points
integration_points = uv_integration(struc)

# coupling
createSolverInterface("WaterLily", "./precice-config.xml", 0, 1)
# dimensions = PreCICE.getDimensions()
writeData = force(sim.body,sim.flow)

vertices_n = Array{Float64,2}(undef, size(u⁰')...)
vertices_f = Array{Float64,2}(undef, size(writeData)...)
vertices_n .= mesh.controlPoints[1:2,:]'
vertices_f[:,1] = integration_points
vertices_f[:,2] .= 0.0

# get mesh ID
ID_n = PreCICE.getMeshID("Nurbs-Mesh-Fluid")
ID_f = PreCICE.getMeshID("Force-Mesh-Fluid")
DataID_n = PreCICE.getDataID("Displacements", ID_n)
DataID_f = PreCICE.getDataID("Forces", ID_f)

# set mesh vertex
vertexIDs_n = PreCICE.setMeshVertices(ID_n, vertices_n)
vertexIDs_f = PreCICE.setMeshVertices(ID_f, vertices_f)

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

# make the writer
wr = vtkWriter("Inverted-Flag"; attrib=custom_attrib)

let # setting local scope for dt outside of the while loop
    dt_precice = PreCICE.initialize()

    # create force and nurbs mesh from the solid data
    # ID_f = PreCICE.getMeshID("Force-Mesh")
    # ID_n = PreCICE.getMeshID("Nurbs-Mesh")
    # n_f = PreCICE.getMeshVertexSize(ID_f)
    # n_d = PreCICE.getMeshVertexSize(ID_n)
    # vertices_f = PreCICE.getMeshVertices(ID_f,Array{Int32}(collect(0:n_f-1)))
    # integration_points = vertices_f[:,1] # uv-components
    # vertices_n = PreCICE.getMeshVertices(ID_n,Array{Int32}(collect(0:n_d-1)))
    # DataID_f = PreCICE.getDataID("Forces", ID_f)
    # DataID_n = PreCICE.getDataID("Displacements", ID_n)
    # vertexIDs_f = PreCICE.setMeshVertices(ID_f, vertices_f)
    # vertexIDs_n = PreCICE.setMeshVertices(ID_n, vertices_n)

    dt = min(0.25, dt_precice)
    PreCICE.writeBlockVectorData(DataID_f, vertexIDs_f, writeData)
    markActionFulfilled(actionWriteInitialData())

    # intialise the coupling
    PreCICE.initializeData()

    # reading initial data
    if PreCICE.isReadDataAvailable()
        # println("WaterLily: Reading initial data")
        readData = PreCICE.readBlockVectorData(DataID_n, vertexIDs_n)
        readData .= u⁰' + readData.*L
        ParametricBodies.update!(body,Matrix(readData'),dt)
    end

    # simulations time
    t = 0.0;

    while PreCICE.isCouplingOngoing()

        # set time step
        dt = min(dt, dt_precice)
        sim.flow.Δt[end] = dt

        if PreCICE.isActionRequired(PreCICE.actionWriteIterationCheckpoint())
            # println("WaterLily: Writing iteration checkpoint")
            store!(store,sim)
            markActionFulfilled(actionWriteIterationCheckpoint())
        end

        if PreCICE.isReadDataAvailable()
            # println("WaterLily: Reading data")
            readData = PreCICE.readBlockVectorData(DataID_n, vertexIDs_n)
            readData .= u⁰' + readData.*L
            ParametricBodies.update!(body,Matrix(readData'),dt)
        end
        
        # solver update
        measure!(sim,t); mom_step!(sim.flow,sim.pois)
        
        if PreCICE.isWriteDataRequired(dt)
            # println("WaterLily: Writing data")
            writeData = force(sim.body,sim.flow)
            if t<=2L # trigger instability
                writeData[:,2] .= -0.5
            end
            # display(writeData)
            PreCICE.writeBlockVectorData(DataID_f, vertexIDs_f, writeData)
        end
        
        dt_precice = PreCICE.advance(dt)

        if PreCICE.isActionRequired(PreCICE.actionReadIterationCheckpoint())
            # println("WaterLily: Reading iteration checkpoint")
            revert!(store,sim)
            markActionFulfilled(actionReadIterationCheckpoint())
        end

        if PreCICE.isTimeWindowComplete()
            t += dt
            # write data
            write!(wr, sim)
        end    
    end # while
    close(wr)
end # let

PreCICE.finalize()
println("WaterLily: Closing Julia solver...")
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

# overwrite the momentum function so that we get the correct BC
@fastmath function WaterLily.mom_step!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC2!(a.u,a.U)
    WaterLily.project!(a,b); BC2!(a.u,a.U)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC2!(a.u,a.U,2)
    WaterLily.project!(a,b,2); a.u ./= 2; BC2!(a.u,a.U)
    push!(a.Δt,WaterLily.CFL(a))
end

# BC function using the profile
function BC2!(a,A,f=1)
    N,n = WaterLily.size_u(a)
    for j ∈ 1:n, i ∈ 1:n
        if i==j # Normal direction, impose profile on inlet and outlet
            for s ∈ (1,2,N[j])
                @WaterLily.loop a[I,i] = f*A[i] over I ∈ WaterLily.slice(N,s,j)
            end
        else  # Tangential directions, interpolate ghost cell to no splip
            @WaterLily.loop a[I,i] = -a[I+δ(j,I),i] over I ∈ WaterLily.slice(N,1,j)
            @WaterLily.loop a[I,i] = -a[I-δ(j,I),i] over I ∈ WaterLily.slice(N,N[j],j)
        end
    end
end

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# make a structure, not actually used
struc = DynamicFEOperator(mesh,gauss_rule,1,1,[],[],ρ=(x)->1; ρ∞=0.0)

## Simulation parameters
L=2^4
Re=500
U=1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[3L,2L].+1.5)
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1); dist=dis);

# make a simulation and a storage
sim = Simulation((4L,6L),(U,0),L;U,ν=U*L/Re,body,ϵ,T=Float64)
store = Store(sim)

# force function
integration_points = uv_integration(struc)

# coupling
createSolverInterface("WaterLily", "./precice-config.xml", 0, 1)
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

let # setting local scope for dt outside of the while loop
    dt_precice = PreCICE.initialize()

    dt = min(0.25, dt_precice)
    PreCICE.writeBlockVectorData(DataID_f, vertexIDs_f, writeData)
    markActionFulfilled(actionWriteInitialData())

    # intialise the coupling
    PreCICE.initializeData()

    # reading initial data
    if PreCICE.isReadDataAvailable()
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
            store!(store,sim)
            markActionFulfilled(actionWriteIterationCheckpoint())
        end

        if PreCICE.isReadDataAvailable()
            readData = PreCICE.readBlockVectorData(DataID_n, vertexIDs_n)
            readData .= u⁰' + readData.*L
            ParametricBodies.update!(body,Matrix(readData'),dt)
        end
        
        # solver update
        measure!(sim,t); mom_step!(sim.flow,sim.pois)
        
        if PreCICE.isWriteDataRequired(dt)
            writeData = force(sim.body,sim.flow)
            PreCICE.writeBlockVectorData(DataID_f, vertexIDs_f, writeData)
        end
        
        dt_precice = PreCICE.advance(dt)

        if PreCICE.isActionRequired(PreCICE.actionReadIterationCheckpoint())
            revert!(store,sim)
            markActionFulfilled(actionReadIterationCheckpoint())
        end

        if PreCICE.isTimeWindowComplete()
            t += dt
        end

    end # while

    get_omega!(sim); plot_vorticity(sim.flow.σ',limit=10)
    # plot!(body.surf, show_cp=false)
    c = [body.surf(s,0.0) for s ∈ 0:0.01:1]
    plot!(getindex.(c,2).+0.5,getindex.(c,1).+0.5,linewidth=2,color=:black,yflip=true)
    savefig("Waterlily_preCICE.png")
end # let

PreCICE.finalize()
println("WaterLily: Closing Julia solver...")
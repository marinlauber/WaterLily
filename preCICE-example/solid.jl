using PreCICE
using Splines
using StaticArrays
using LinearAlgebra
# structure to store solid state
struct Store
    dˢ::AbstractArray
    vˢ::AbstractArray
    aˢ::AbstractArray
    function Store(struc::AbstractFEOperator)
        new(copy(struc.u[1]),copy(struc.u[2]),copy(struc.u[3]))
    end
end
function store!(s::Store,struc::AbstractFEOperator)
    s.dˢ .= struc.u[1];
    s.vˢ .= struc.u[2]
    s.aˢ .= struc.u[3];
end
function revert!(s::Store,struc::AbstractFEOperator)
    struc.u[1] .= s.dˢ;
    struc.u[2] .= s.vˢ;
    struc.u[3] .= s.aˢ;
end
# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
EI = 0.35
EA = 100_000.0
density(ξ) = 0.5

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptRight, 0.0; comp=1),
    Boundary1D("Neumann", ptRight, 0.0; comp=2)
]

# make a structure and a storage
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC, ρ=density; ρ∞=0.0)
store = Store(struc)

# coupling
PreCICE.createParticipant("Splines", "./precice-config.xml", 0, 1)

dimensions = PreCICE.getMeshDimensions("Nurbs-Mesh-Solid")
writeData = zeros(size(mesh.controlPoints[1:2,:]'))

# location of integration points
integration_points = uv_integration(struc)

vertices_nurbs = Array{Float64,2}(undef, size(mesh.controlPoints[1:2,:]'))
vertices_forces = Array{Float64,2}(undef, length(integration_points), dimensions)
vertices_knots = Array{Float64,2}(undef, length(mesh.knots), dimensions)
vertices_nurbs .= mesh.controlPoints[1:2,:]'
vertices_forces[:,1] .= integration_points[:] # the mesh is only defined in the parametric spaces
vertices_forces[:,2] .= 0.0
vertices_knots[:,1] .= mesh.knots
vertices_knots[:,2] .= 1:length(mesh.knots)

# set mesh vertex and give access to precice
vertexIDs_nurbs = PreCICE.setMeshVertices("Nurbs-Mesh-Solid", vertices_nurbs)
vertexIDs_forces = PreCICE.setMeshVertices("Force-Mesh-Solid", vertices_forces)
vertexIDs_knots = PreCICE.setMeshVertices("Knots-Mesh", vertices_knots)

# wierdly this allows to access both meshes...
PreCICE.setMeshAccessRegion("Nurbs-Mesh-Solid",[-1. 1.; -1. 1.])

let # setting local scope for dt outside of the while loop
    
    L = 2^5 # needed from the fluid for scaling
    dt = 0.5

    # start coupling
    PreCICE.initialize()

    # start sim
    t = 0.0

    while PreCICE.isCouplingOngoing()

        dt_precice = PreCICE.getMaxTimeStepSize()
        dt = min(dt, dt_precice)

        if PreCICE.requiresWritingCheckpoint()
            store!(store,struc)
        end

        readData = PreCICE.readData("Force-Mesh-Solid", "Forces", vertexIDs_forces, dt)
        @show readData
        
        # update the structure
        solve_step!(struc, Matrix(readData'), dt/L)

        # iwrite the data to the mesh
        writeData .= points(struc)'
        PreCICE.writeData("Nurbs-Mesh-Solid", "Displacements", vertexIDs_nurbs, writeData)

        dt_precice = PreCICE.advance(dt)

        if PreCICE.requiresReadingCheckpoint()
            revert!(store,struc)
        else
            t += dt
        end
    end # while

end # let

PreCICE.finalize()
println("Splines: Closing Julia solver...")
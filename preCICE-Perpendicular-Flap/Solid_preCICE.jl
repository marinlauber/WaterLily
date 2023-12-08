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
EI = 4.0
EA = 400000.0
density(ξ) = 3

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
createSolverInterface("Splines", "./precice-config.xml", 0, 1)

dimensions = PreCICE.getDimensions()
writeData = 0.0*Matrix(mesh.controlPoints[1:2,:]')

# location of integration points
integration_points = Splines.uv_integration(struc)

vertices_n = Array{Float64,2}(undef, size(mesh.controlPoints[1:2,:]'))
vertices_f = Array{Float64,2}(undef, length(integration_points), dimensions)
vertices_n .= mesh.controlPoints[1:2,:]'
vertices_f[:,1] .= integration_points[:] # the mesh is only defined in the parametric spaces
vertices_f[:,2] .= 0.0


# get mesh ID
ID_n = PreCICE.getMeshID("Nurbs-Mesh-Solid")
ID_f = PreCICE.getMeshID("Force-Mesh-Solid")
DataID_n = PreCICE.getDataID("Displacements", ID_n)
DataID_f = PreCICE.getDataID("Forces", ID_f)

# set mesh vertex
vertexIDs_n = PreCICE.setMeshVertices(ID_n, vertices_n)
vertexIDs_f = PreCICE.setMeshVertices(ID_f, vertices_f)

let # setting local scope for dt outside of the while loop

    # start coupling
    dt_precice = PreCICE.initialize()

    L = 2^4 # needed from the fluid for scaling
    dt = min(0.25, dt_precice)
    PreCICE.writeBlockVectorData(DataID_n, vertexIDs_n, writeData)
    markActionFulfilled(actionWriteInitialData())

    # intialise the coupling
    PreCICE.initializeData()

    # reading initial data
    if PreCICE.isReadDataAvailable()
        readData = PreCICE.readBlockVectorData(DataID_f, vertexIDs_f)
    end

    # start time
    t = 0.0;

    while PreCICE.isCouplingOngoing()

        if PreCICE.isActionRequired(PreCICE.actionWriteIterationCheckpoint())
            store!(store,struc)
            markActionFulfilled(actionWriteIterationCheckpoint())
        end

        if PreCICE.isReadDataAvailable()
            readData = PreCICE.readBlockVectorData(DataID_f, vertexIDs_f)
        end

        # update the structure
        solve_step!(struc, Matrix(readData'), dt/L)
        
        if PreCICE.isWriteDataRequired(dt)
            writeData .= reshape(struc.u[1][1:2mesh.numBasis],(mesh.numBasis,2))
            PreCICE.writeBlockVectorData(DataID_n, vertexIDs_n, writeData)
        end

        dt_precice = PreCICE.advance(dt)

        if PreCICE.isActionRequired(PreCICE.actionReadIterationCheckpoint())
            revert!(store,struc)
            markActionFulfilled(actionReadIterationCheckpoint())
        end

        if PreCICE.isTimeWindowComplete()
            t += dt
        end

    end # while

end # let

PreCICE.finalize()
println("Splines: Closing Julia solver...")
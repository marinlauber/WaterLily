using PreCICE
using Splines
using StaticArrays
using LinearAlgebra

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
EI = 4.0
EA = 400000.0
density(ξ) = 3.0

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

# make a structure
struc = GeneralizedAlpha(FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC; ρ=density); ρ∞=0.5)


# coupling
createSolverInterface("Splines", "./precice-config.xml", 0, 1)

dimensions = PreCICE.getDimensions()
numberOfVertices = 3
# zero displacements
writeData = 0.0*Matrix(mesh.controlPoints[1:2,:]')

# location of integration points
integration_points = Splines.uv_integration(struc.op)

vertices_n = Array{Float64,2}(undef, size(mesh.controlPoints[1:2,:]'))
vertices_f = Array{Float64,2}(undef, length(integration_points), dimensions)
vertices_n .= mesh.controlPoints[1:2,:]'
vertices_f[:,1] .= integration_points[:]
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

    L = 2^4
    dt = min(0.25, dt_precice)
    PreCICE.writeBlockVectorData(DataID_n, vertexIDs_n, writeData)
    markActionFulfilled(actionWriteInitialData())

    # intialise the coupling
    PreCICE.initializeData()

    # reading initial data
    if PreCICE.isReadDataAvailable()
        # println("Splines: Reading initial data")
        readData = PreCICE.readBlockVectorData(DataID_f, vertexIDs_f)
    end

    t = 0.0
    cache = (copy(struc.u[1]),copy(struc.u[2]),copy(struc.u[3]))

    while PreCICE.isCouplingOngoing()

        dt = min(dt, dt_precice)

        if PreCICE.isActionRequired(PreCICE.actionWriteIterationCheckpoint())
            # println("Splines: Writing iteration checkpoint")
            cache = (copy(struc.u[1]),copy(struc.u[2]),copy(struc.u[3]))
            markActionFulfilled(actionWriteIterationCheckpoint())
        end

        if PreCICE.isReadDataAvailable()
            # println("Splines: Reading data")
            readData = PreCICE.readBlockVectorData(DataID_f, vertexIDs_f)
            # display(readData)
        end

        # update the structure
        solve_step!(struc, Matrix(readData'), dt/L)

        if PreCICE.isWriteDataRequired(dt)
            # println("Splnies: Writing data")
            writeData .= reshape(struc.u[1][1:2mesh.numBasis],(mesh.numBasis,2))
            # display(writeData)
            PreCICE.writeBlockVectorData(DataID_n, vertexIDs_n, writeData)
        end

        dt_precice = PreCICE.advance(dt)

        if PreCICE.isActionRequired(PreCICE.actionReadIterationCheckpoint())
            # println("Splines: Reading iteration checkpoint")
            struc.u[1] .= cache[1]
            struc.u[2] .= cache[2]
            struc.u[3] .= cache[3]
            markActionFulfilled(actionReadIterationCheckpoint())
        end

        if PreCICE.isTimeWindowComplete()
            t += dt
        end

    end # while

end # let

PreCICE.finalize()
println("Splines: Closing Julia solver...")
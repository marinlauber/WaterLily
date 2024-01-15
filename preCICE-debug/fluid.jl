using PreCICE

commRank = 0
commSize = 1

# FLUID SOLVER
solverName = "Fluid"
configFileName = "precice-config-2.xml"
meshName = "Nurbs-Mesh"
meshNameTwin = "Forces-Mesh"
dataWriteName = "Displacements"
dataReadName = "Forces"

println(
    """DUMMY: Running solver dummy with preCICE config file "$configFileName", participant name "$solverName", and mesh name "$meshName" """,
)
PreCICE.createSolverInterface(solverName, configFileName, commRank, commSize)

# dimensions = PreCICE.getDimensions()

# numberOfVertices = 3

# writeData = zeros(numberOfVertices, dimensions)

# vertices = Array{Float64,2}(undef, numberOfVertices, dimensions)

# for i = 1:numberOfVertices, j = 1:dimensions
#     vertices[i, j] = i
# end

# meshID = PreCICE.getMeshID(meshName)

# vertexIDs = PreCICE.setMeshVertices(meshID, vertices)

# dataWriteID = PreCICE.getDataID(dataWriteName, meshID)

# dataReadID = PreCICE.getDataID(dataReadName, meshID)

# # try get the other solver's mesh vertex
# # meshIDTwin = PreCICE.getMeshID(meshNameTwin)
# # println("MeshIDTwin: ", meshIDTwin)
# # twin_vertex_size = PreCICE.getMeshVertexSize(meshIDTwin)
# # println("Twin vertex size: ", twin_vertex_size)
# # vertices_twin = PreCICE.getMeshVertices(twin_vertex_size,Array{Int32}(collect(0:twin_vertex_size-1)))
# # println("Twin vertices: ", vertices_twin)
# let # setting local scope for dt outside of the while loop

#     dt = PreCICE.initialize()

#     while PreCICE.isCouplingOngoing()

#         # if PreCICE.requiresWritingCheckpoint()
#             # println("DUMMY: Writing iteration checkpoint")
#         # end
#         if PreCICE.isActionRequired(PreCICE.actionWriteIterationCheckpoint())
#             println("DUMMY: Writing iteration checkpoint")
#             markActionFulfilled(actionWriteIterationCheckpoint())
#         end

#         # dt = PreCICE.getMaxTimeStepSize()
#         # readData = PreCICE.readData(meshName, dataReadName, vertexIDs, dt)
#         readData = PreCICE.readBlockVectorData(dataReadID, vertexIDs)

#         for i = 1:numberOfVertices, j = 1:dimensions
#             writeData[i, j] = readData[i, j] + 1.0
#         end

#         # PreCICE.writeData(meshName, dataWriteName, vertexIDs, writeData)
#         PreCICE.writeBlockVectorData(dataWriteID, vertexIDs, writeData)

#         dt = PreCICE.advance(dt)

#         # if PreCICE.requiresReadingCheckpoint()
#             # println("DUMMY: Reading iteration checkpoint")
#         if PreCICE.isActionRequired(PreCICE.actionReadIterationCheckpoint())
#             println("DUMMY: Reading iteration checkpoint")
#             markActionFulfilled(actionReadIterationCheckpoint())
#         else
#             println("DUMMY: Advancing in time")
#         end

#     end # while

# end # let

# PreCICE.finalize()
println("DUMMY: Closing Julia solver dummy...")
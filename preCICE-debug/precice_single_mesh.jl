using PreCICE
# void Participant::getMeshVertexIDsAndCoordinates(::precice::string_view    meshName,
#                                                  ::precice::span<VertexID> ids,
#                                                  ::precice::span<double>   coordinates) const
function getMeshVertexIDsAndCoordinates(
    meshName::String,
)::Tuple{AbstractArray{Integer},AbstractArray{Float64}}
    @warn "The function getMeshVertexIDsAndCoordinates is still experimental"

    _size = getMeshVertexSize(meshName)
    vertexIDs = zeros(Cint, _size)
    vertexCoordinates = zeros(Float64, _size * getMeshDimensions(meshName))
    ccall(
        (:precicec_getMeshVertexIDsAndCoordinates, "libprecice"),
        Cvoid,
        (Ptr{Int8}, Cint, Ref{Cint}, Ref{Cdouble}),
        meshName,
        _size,
        vertexIDs,
        vertexCoordinates,
    )
    return vertexIDs,reshape(vertexCoordinates, (_size, getMeshDimensions(meshName)))
end

commRank = 0
commSize = 1

if size(ARGS, 1) < 1
    @error(
        "ERROR: pass config path and solver name, example: julia solverdummy.jl ./precice-config.xml SolverOne",
    )
    exit(1)
end

configFileName = ARGS[1]
solverName = ARGS[2]

# the both have the same mesh now
meshName = "SolverOne-Mesh"

if solverName == "SolverOne"
    dataWriteName = "Data-One"
    dataReadName = "Data-Two"
else
    dataReadName = "Data-One"
    dataWriteName = "Data-Two"
end


println(
    """DUMMY: Running solver dummy with preCICE config file "$configFileName", participant name "$solverName", and mesh name "$meshName" """,
)
PreCICE.createParticipant(solverName, configFileName, commRank, commSize)

let # setting local scope for dt outside of the while loop

    # only SolverOne has a mesh, but he must share it
    if solverName=="SolverOne"
        dimensions = PreCICE.getMeshDimensions(meshName)
        numberOfVertices = 8
        vertices = Array{Float64,2}(undef, numberOfVertices, dimensions)
        for i = 1:numberOfVertices
            vertices[i, :] = [i/numberOfVertices*2-1,0.] # spaced in [-1,1]
        end
        vertexIDs = PreCICE.setMeshVertices(meshName, vertices)
        PreCICE.setMeshAccessRegion(meshName,[-1. 1.; -1. 1.])
    end
    
    PreCICE.initialize()
    
    # SolverTwo needs to get the mesh from SolverOne
    if solverName=="SolverTwo"
        (vertexIDs, vertices) = getMeshVertexIDsAndCoordinates(meshName)
        vertices = Array{Float64,2}(vertices)
        vertexIDs = Array{Int32}(vertexIDs)
    end

    # we can check that they both now have the same mesh
    @show solverName
    @show vertexIDs, vertices
    @show size(vertices)
    
    # now we can generate the write data, we cannot do that before because 
    # we need to know the number of vertices
    numberOfVertices, dimensions = size(vertices)
    writeData = zeros(numberOfVertices, dimensions)

    # standard from now
    while PreCICE.isCouplingOngoing()

        if PreCICE.requiresWritingCheckpoint()
            println("DUMMY: Writing iteration checkpoint")
        end

        dt = PreCICE.getMaxTimeStepSize()
        readData = PreCICE.readData(meshName, dataReadName, vertexIDs, dt)
        @show readData

        for i = 1:numberOfVertices, j = 1:dimensions
            writeData[i, j] = readData[i, j] + 1.0
        end

        @show writeData
        PreCICE.writeData(meshName, dataWriteName, vertexIDs, writeData)

        PreCICE.advance(dt)

        if PreCICE.requiresReadingCheckpoint()
            println("DUMMY: Reading iteration checkpoint")
        else
            println("DUMMY: Advancing in time")
        end

    end # while

end # let

PreCICE.finalize()
println("DUMMY: Closing Julia solver dummy...")

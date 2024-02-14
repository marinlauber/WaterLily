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
    return vertexIDs,
    permutedims(reshape(vertexCoordinates, (_size, getMeshDimensions(meshName))))
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

if solverName == "SolverOne"
    meshName = "SolverOne-Mesh"
    meshOther = "SolverTwo-Mesh"
    dataWriteName = "Data-One"
    dataReadName = "Data-Two"
else
    meshName = "SolverTwo-Mesh"
    meshOther = "SolverOne-Mesh"
    dataReadName = "Data-One"
    dataWriteName = "Data-Two"
end


println(
    """DUMMY: Running solver dummy with preCICE config file "$configFileName", participant name "$solverName", and mesh name "$meshName" """,
)
PreCICE.createParticipant(solverName, configFileName, commRank, commSize)

dimensions = PreCICE.getMeshDimensions(meshName)

numberOfVertices = 3

writeData = zeros(numberOfVertices, dimensions)

vertices = Array{Float64,2}(undef, numberOfVertices, dimensions)

for i = 1:numberOfVertices, j = 1:dimensions
    vertices[i, j] = 2rand()-1 # spaced in [-1,1]
end

vertexIDs = PreCICE.setMeshVertices(meshName, vertices)

let # setting local scope for dt outside of the while loop
    
    PreCICE.setMeshAccessRegion(meshName,[-1. 1.; -1. 1.])
    PreCICE.initialize()

    # get the coordinates of the nodses
    (IDother, CoordsOther) = getMeshVertexIDsAndCoordinates(meshOther)
    println("my coordinates")
    @show vertexIDs, vertices
    println("the other's coordinates")
    @show IDother, CoordsOther

    while PreCICE.isCouplingOngoing()

        if PreCICE.requiresWritingCheckpoint()
            println("DUMMY: Writing iteration checkpoint")
        end

        dt = PreCICE.getMaxTimeStepSize()
        readData = PreCICE.readData(meshName, dataReadName, vertexIDs, dt)

        for i = 1:numberOfVertices, j = 1:dimensions
            writeData[i, j] = readData[i, j] + 1.0
        end

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

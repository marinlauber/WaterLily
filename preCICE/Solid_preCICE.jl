using PreCICE
using Splines
using StaticArrays
using LinearAlgebra

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
A = 0.1
L = 1.0
EI = 0.25
EA = 10000.0
f(s) = [0.0,0.0] # s is curvilinear coordinate

# natural frequencies
ωₙ = 1.875; fhz = 0.125
density(ξ) = (ωₙ^2/2π)^2*(EI/(fhz^2*L^4))

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

# make a problem
p = EulerBeam(EI, EA, f, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)

## Time integration
ρ∞ = 0.5; # spectral radius of the amplification matrix at infinitely large time step
αm = (2.0 - ρ∞)/(ρ∞ + 1.0);
αf = 1.0/(1.0 + ρ∞)
γ = 0.5 - αf + αm;
β = 0.25*(1.0 - αf + αm)^2;
# unconditional stability αm ≥ αf ≥ 1/2

# coupling
createSolverInterface("Splines", "./precice-config.xml", 0, 1)

dimensions = PreCICE.getDimensions()
numberOfVertices = 3
writeData = Matrix(mesh.controlPoints[1:2,:]')

# location of integration points
integration_points = Splines.uv_integration(p)

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

    # unpack variables
    @unpack x, resid, jacob = p
    M = spzero(jacob)
    stiff = zeros(size(jacob))
    fext = zeros(size(resid)); loading = zeros(size(resid))
    M = global_mass!(M, mesh, density, gauss_rule)
    a0 = zeros(size(resid))
    dⁿ = u₀ = zero(a0);
    vⁿ = zero(a0);
    aⁿ = zero(a0);

    # start coupling
    PreCICE.initialize()

    L = 2^4
    dt = 0.2
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
    cache = (dⁿ, vⁿ, aⁿ)

    while PreCICE.isCouplingOngoing()

        if PreCICE.isActionRequired(PreCICE.actionWriteIterationCheckpoint())
            # println("Splines: Writing iteration checkpoint")
            cache = (dⁿ, vⁿ, aⁿ)
            markActionFulfilled(actionWriteIterationCheckpoint())
        end

        if PreCICE.isReadDataAvailable()
            # println("Splines: Reading data")
            readData = PreCICE.readBlockVectorData(DataID_f, vertexIDs_f)
            # display(readData)
        end

        # update the structure
        dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹ = Splines.step2(jacob, stiff, Matrix(M), resid, fext,
                                         Matrix(readData'), dⁿ, vⁿ, aⁿ, t/L, (t+dt)/L, αm, αf, β, γ, p)
        
        if PreCICE.isWriteDataRequired(dt)
            # println("Splnies: Writing data")
            writeData .= reshape(dⁿ⁺¹[1:2p.mesh.numBasis],(p.mesh.numBasis,2))
            PreCICE.writeBlockVectorData(DataID_n, vertexIDs_n, writeData)
        end

        PreCICE.advance(dt)

        if PreCICE.isActionRequired(PreCICE.actionReadIterationCheckpoint())
            # println("Splines: Reading iteration checkpoint")
            dⁿ, vⁿ, aⁿ = cache
            markActionFulfilled(actionReadIterationCheckpoint())
        end

        if PreCICE.isTimeWindowComplete()
            dⁿ.= dⁿ⁺¹
            vⁿ.= vⁿ⁺¹
            aⁿ.= aⁿ⁺¹
            t += dt
        end

    end # while

end # let

PreCICE.finalize()
println("Splines: Closing Julia solver...")
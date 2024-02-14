using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
include("TwoD_plots.jl")
include("../src/Coupling.jl")

function force(b::DynamicBody,flow::Flow)
    reduce(hcat,[NurbsForce(b.surf,flow.p,s) for s ∈ integration_points])
end

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0

# parameters
EI = 0.35         # Cauhy number
EA = 100_000.0  # make inextensible
density(ξ) = 0.5  # mass ratio

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
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, 
                         Dirichlet_BC, Neumann_BC, ρ=density; ρ∞=0.0)

## Simulation parameters
L=2^4
Re=200
U=1
ϵ=0.5
thk=2ϵ+√2

# spline distance function
dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:].*L.+[2L,3L].+1.5)
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1); dist=dis);

# force function
integration_points = uv_integration(struc)

# make a coupled sim
sim = CoupledSimulation((8L,6L),(U,0),L,body,struc,IQNCoupling;
                         U,ν=U*L/Re,ϵ,ωᵣ=0.05,maxCol=6,T=Float64)

# sime time
t₀ = round(sim_time(sim)); duration = 10.0; step = 0.2
iterations = []
# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step)
    
    # sim_step!(sim,tᵢ)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ*sim.L/sim.U
        
        println("  tᵢ=$tᵢ, t=$(round(t,digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        store!(sim); iter=1;

        # iterative loop
        while true

            #  integrate once in time
            solve_step!(sim.struc, sim.forces, sim.flow.Δt[end]/sim.L)
            
            # update flow, this requires scaling the displacements
            ParametricBodies.update!(sim.body,u⁰.+L*sim.pnts,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)

            # get new coupling variable
            sim.pnts .= points(sim.struc)
            sim.forces .= force(sim.body,sim.flow)

            if t/sim.L*sim.U<2.0
                sim.forces[2,:] .-= 0.5 # add vertical force
            end

            # accelerate coupling
            print("    iteration: ",iter)
            converged = update!(sim.cpl, sim.pnts, sim.forces, 0.0)

            # check for convengence
            (converged || iter+1 > 15) && break

            # if we have not converged, we must revert
            revert!(sim); iter += 1
        end
        push!(iterations,iter)
        println(" beam length: ", integrate(sim.body.surf))
        # finish the time step
        t += sim.flow.Δt[end]
    end


    # println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body.surf)
    plot!(title="tU/L $tᵢ")
end

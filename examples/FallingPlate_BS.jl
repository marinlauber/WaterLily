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

function mean(p::AbstractArray;dims=1)
    sum(p,dims=dims)/size(p,dims)
end

# Material properties and mesh
numElem=10
degP=3
ptLeft = 0.0
ptRight = 1.0

# parameters
EI = 0.001    # Cauhy number
EA = 50_000.0  # make inextensible
density(ξ) = 1.0  # mass ratio
B = 100

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# make a structure
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, 
                          [], [], ρ=density; ρ∞=0.0)

## Simulation parameters
L=2^5
Re=200
U=1
ϵ=0.5
thk=2ϵ+√2

# stiffness parameters
Fg = B*EI/L^2

# spline distance function
dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:].*L.+[1.5L,L].+1.5)
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1); dist=dis);

# force function
integration_points = uv_integration(struc)

# make a coupled sim
global U_cm = SA[0.,0.]
global X_cm = SA[0.,0.]
Ut(i,t) = -U_cm[i]
sim = CoupledSimulation((4L,4L),Ut,L,body,struc,IQNCoupling;
                         U,ν=U*L/Re,ϵ,ωᵣ=0.05,maxCol=12,T=Float64)

# sime time
t₀ = round(sim_time(sim)); duration = 10.0; step = 0.2
iterations = []
# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step)
    
    # update until time tᵢ in the background
    while WaterLily.time(sim) < tᵢ*sim.L/sim.U
        
        println("  t=$(round(WaterLily.time(sim),digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        store!(sim); iter=1;

        # get position and velocity of CM of structure
        global X_cm = SA[mean(points(sim.struc).*sim.L;dims=2)...]
        global U_cm = SA[mean(vⁿ(sim.struc);dims=2)...]
        @show X_cm, U_cm

        # iterative loop
        while true

            # integrate once in time
            solve_step!(sim.struc, sim.forces, sim.flow.Δt[end]/sim.L)
            
            # update flow, this requires scaling the displacements
            ParametricBodies.update!(sim.body,u⁰.+(sim.L*sim.pnts.-X_cm),sim.flow.Δt[end])
            measure!(sim); mom_step!(sim.flow,sim.pois)

            # get new coupling variable
            sim.pnts .= points(sim.struc)
            sim.forces .= force(sim.body,sim.flow)

            # gravity force
            sim.forces[2,:] .-= 0.5 # add vertical gravity force

            # accelerate coupling
            print("    iteration: ",iter)
            converged = update!(sim.cpl, sim.pnts, sim.forces, 0.0)

            # check for convengence
            (converged || iter+1 > 20) && break

            # if we have not converged, we must revert
            revert!(sim); iter += 1
        end
        push!(iterations,iter)
    end

    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body.surf;add_cp=false)
    plot!(title="tU/L $tᵢ")

    # check that we are still inside the domain
    pos = mean(sim.body.surf.pnts;dims=2)
    !(all(pos.>[0.,0.]) && all(pos.<size(sim.flow.p))) && break
end
using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
# using SparseArrays
include("examples/TwoD_plots.jl")
include("Coupling.jl")

# function force(b::DynamicBody,sim::Simulation)
#     reduce(hcat,[NurbsForce(b.surf,sim.flow.p,s) for s ∈ integration_points])
# end
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
density(ξ) = 0.3  # mass ratio

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
struc = GeneralizedAlpha(FEOperator(mesh, gauss_rule, EI, EA, 
                         Dirichlet_BC, Neumann_BC; ρ=density);
                         ρ∞=0.0)

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
integration_points = Splines.uv_integration(struc.op)

# intialise coupling+
f_old = zeros((2,length(integration_points)))
pnts_old = zero(u⁰)

# set up coupling
# QNCouple = Relaxation(points(struc),f_old;relax=0.8)
QNCouple = IQNCoupling(points(struc),f_old;relax=0.05,maxCol=6)
updated_values = zero(QNCouple.x)

sim = CoupledSimulation((8L,6L),(U,0),L,body,struc,QNCouple;U,ν=U*L/Re,ϵ,T=Float64)

t₀ = round(sim_time(sim))
duration = 30.0
step = 0.2

# time loop
@gif for tᵢ in range(t₀,t₀+duration;step)

    global f_old, pnts_old, updated_values;

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])

    while t < tᵢ*sim.L/sim.U
        
        println("  tᵢ=$tᵢ, t=$(round(t,digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        store!(sim)
        
        # implicit solve
        iter=1; firstIteration=true

        # iterative loop
        while true

            #  integrate once in time
            solve_step!(sim.struc, f_old, sim.flow.Δt[end]/sim.L)
            pnts_new = points(sim.struc)
            
            # update flow, this requires scaling the displacements
            ParametricBodies.update!(sim.body,u⁰.+L*pnts_old,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            f_new = force(sim.body,sim.flow)

            if t/sim.L*sim.U<2.0
                f_new .-= 0.5
            end

            # accelerate coupling
            concatenate!(updated_values, pnts_new, f_new, sim.cpl.subs)
            res_comb = res(updated_values, sim.cpl.x)
            println("    iteration: ",iter," r₂: ",res_comb, " converged: : ",res_comb<1e-2)
            converged = update!(sim.cpl, updated_values, firstIteration)
            revert!(updated_values, pnts_old, f_old, sim.cpl.subs)
            if converged || iter+1 > 50 
                break
            end

            # if we have not converged, we must revert
            revert!(sim)
            iter += 1
        end

        # finish the time step
        t += sim.flow.Δt[end]
    end

    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body.surf)
    plot!(title="tU/L $tᵢ")
end

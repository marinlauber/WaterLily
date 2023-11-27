using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
# using SparseArrays
include("examples/TwoD_plots.jl")
include("Coupling.jl")

function force(b::DynamicBody,flow::Flow)
    reduce(hcat,[NurbsForce(b.surf,flow.p,s) for s ∈ integration_points])
end

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
L=1.0
EI = 0.35
EA = 100000.0
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

# make a structure
struc = GeneralizedAlpha(FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC; ρ=density); ρ∞=0.0)

## Simulation parameters
L=2^4
Re=200
U=1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:].*L.+[3L,2L].+1.5)
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1));
flow = Flow((6L,4L),(U,0);Δt=0.25,ν=U*L/Re,T=Float64)
poiss = MultiLevelPoisson(flow.p,flow.μ₀,flow.σ)

# intialise coupling
integration_points = Splines.uv_integration(struc.op)
forces = force(body,flow);
pnts = zero(u⁰);
Coupling = Relaxation(dⁿ(struc),f_old;relax=0.8)

sim = CoupledSimulation(U,L,ϵ,flow,body,poiss,struc,QNCouple;T=Float64)

t₀ = round(sim_time(sim))
duration =1.0
tstep = 0.1


# time loop
@gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    global forces, pnts
    
    # update coupled sim
    # sim_step!(sim,tᵢ;verbose=true)

    t = sim_time(sim)
    while t < tᵢ*sim.L/sim.U
        store!(sim); iter=1
        while iter < 50
            # update structure
            solve_step!(sim.stru,forces,sim.flow.Δt[end]/sim.L)
            # update body
            ParametricBodies.update!(sim.body,pnts,sim.flow.Δt[end])
            # update flow
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            # check convergence and accelerate
            forces=force(sim.body,sim.flow); pnts=points(sim.struc)
             # accelerate coupling
            concatenate!(xᵏ, pnts, forces, QNCouple.subs)
            res_comb = res(updated_values, QNCouple.x)
            println("    iteration: ",iter," r₂: ",res_comb, " converged: ",res_comb<1e-2)
            converged = update!(sim.coupling,xᵏ,Val(iter==1))
            revert!(xᵏ, pnts, forces, QNCouple.subs)
            converged && break
            # revert if not convergend
            revert!(sim); iter+=1
        end
    end
    
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    # get_omega!(sim);plot_vorticity(sim.flow.σ, limit=10)
    flood(sim.flow.p[inside(sim.flow.p)], clims=(-1,1))
    plot!(body.surf)
    plot!(title="tU/L $tᵢ")
end

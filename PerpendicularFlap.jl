using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
include("examples/TwoD_plots.jl")
include("Coupling.jl")

function force(b::DynamicBody,sim::Simulation)
    reduce(hcat,[ParametricBodies.NurbsForce(b.surf,sim.flow.p,s) for s ∈ integration_points])
end

# ENV["JULIA_DEBUG"] = Main

# overwrite the momentum function so that we get the correct BC
@fastmath function WaterLily.mom_step!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC2!(a.u,a.U)
    WaterLily.project!(a,b); BC2!(a.u,a.U)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC2!(a.u,a.U,2)
    WaterLily.project!(a,b,2); a.u ./= 2; BC2!(a.u,a.U)
    push!(a.Δt,WaterLily.CFL(a))
end

# BC function using the profile
function BC2!(a,A,f=1)
    N,n = WaterLily.size_u(a)
    for j ∈ 1:n, i ∈ 1:n
        if i==j # Normal direction, impose profile on inlet and outlet
            for s ∈ (1,2,N[j])
                @WaterLily.loop a[I,i] = f*A[i] over I ∈ WaterLily.slice(N,s,j)
            end
        else  # Tangential directions, interpolate ghost cell to no splip
            @WaterLily.loop a[I,i] = -a[I+δ(j,I),i] over I ∈ WaterLily.slice(N,1,j)
            @WaterLily.loop a[I,i] = -a[I-δ(j,I),i] over I ∈ WaterLily.slice(N,N[j],j)
        end
    end
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

# make a structure
struc = GeneralizedAlpha(FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC; ρ=density); ρ∞=0.0)

## Simulation parameters
L=2^4
Re=100
U=1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[3L,3L].+1.5)
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1));

# make a simulation
sim = Simulation((4L,6L), (0,U), L; ν=U*L/Re, body, T=Float64)
# sim.flow.Δt[end] = 0.2

# duration of the simulation
duration = 10.0
step = 0.1
t₀ = 0.0
ωᵣ = 0.5 # ωᵣ ∈ [0,1] is the relaxation parameter

# force functions
integration_points = Splines.uv_integration(struc.op)

# intialise coupling
f_old = force(body,sim); size_f = size(f_old)
pnts_old = zero(u⁰)

# QNCouple = Relaxation(dⁿ(struc),f_old;relax=0.9)
QNCouple = IQNCoupling(dⁿ(struc),f_old;relax=ωᵣ,maxCol=4)
updated_values = zero(QNCouple.x)

@time @gif for tᵢ in range(t₀,t₀+duration;step)

    global f_old, pnts_old, updated_values;

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])

    while t < tᵢ*sim.L/sim.U
        
        println("  tᵢ=$tᵢ, t=$(round(t,digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        WaterLily.store!(sim.flow);
        ParametricBodies.store!(sim.body);
        cache = (copy(struc.u[1]),copy(struc.u[2]),copy(struc.u[3]))
        
        # time steps
        Δt = sim.flow.Δt[end]/sim.L*sim.U
        tⁿ = t/sim.L*sim.U; # previous time instant
        
        # implicit solve
        iter=1; firstIteration=true

        # iterative loop
        while true

            #  integrate once in time
            solve_step!(struc, f_old, Δt)
            pnts_new = dⁿ(struc)
            
            # update flow, this requires scaling the displacements
            ParametricBodies.update!(body,u⁰.+L*pnts_old,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            # sim.flow.Δt[end] = 0.2
            f_new = force(body,sim)

            # check that residuals have converged
            rd = res(pnts_old,pnts_new); rf = res(f_old,f_new);
            println("    Iter: ",iter,", rd: ",round(rd,digits=8),", rf: ",round(rf,digits=8))
            if ((rd<1e-2) && (rf<1e-2)) || iter+1 > 50 # if we converge, we exit to avoid reverting the flow
                println("  Converged...")
                # if time step converged, reset coupling preconditionner
                concatenate!(updated_values, pnts_new, f_new, QNCouple.subs)
                finalize!(QNCouple, updated_values)
                f_old .= f_new; pnts_old .= pnts_new
                firstIteration = true
                break
            end

            # accelerate coupling
            concatenate!(updated_values, pnts_new, f_new, QNCouple.subs)
            updated_values = update(QNCouple, updated_values, firstIteration)
            revert!(updated_values, pnts_old, f_old, QNCouple.subs)
        
            # if we have not converged, we must revert
            WaterLily.revert!(sim.flow)
            ParametricBodies.revert!(sim.body);
            struc.u[1] .= cache[1]
            struc.u[2] .= cache[2]
            struc.u[3] .= cache[3]
            iter += 1
            firstIteration = false
        end

        # finish the time step
        t += sim.flow.Δt[end]
    end

    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim); plot_vorticity(sim.flow.σ', limit=10)
    c = [body.surf(s,0.0) for s ∈ 0:0.01:1]
    plot!(getindex.(c,2).+0.5,getindex.(c,1).+0.5,linewidth=2,color=:black,yflip = true)
    plot!(title="tU/L $tᵢ")
end

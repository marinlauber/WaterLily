using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
using SparseArrays
include("examples/TwoD_plots.jl")
include("Coupling.jl")

function force(b::DynamicBody,sim::Simulation)
    reduce(hcat,[ParametricBodies.NurbsForce(b.surf,sim.flow.p,s) for s ∈ integration_points])
end

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
L=2^5
Ca = 0.35
Mᵨ = 10
nu = 0.3
h = L/100
E = Ca*L^3*12*(1-nu^2)/h^3
EI = E*h^3/12
EA = 1.0e6
f(s) = [0.0,0.0] # s is curvilinear coordinate
density(s) = Mᵨ*L/h

# # natural frequencies
# ωₙ = 1.875; fhz = 0.125
# density(ξ) = (ωₙ^2/2π)^2*(EI/(fhz^2*L^4))
# println(ωₙ^2.0*√(EI/(density(0.5)*L^4))/(2π))

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

# unpack variables
@unpack x, resid, jacob = p
M = spzero(jacob)
stiff = zeros(size(jacob))
fext = zeros(size(resid))
M = global_mass!(M, mesh, density, gauss_rule)

# initialise
a0 = zeros(size(resid))
dⁿ = u₀ = zero(a0);
vⁿ = zero(a0);
aⁿ = zero(a0);

## Simulation parameters
L=2^5
Re=500
U=1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[L,2L])
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1));
sim = Simulation((6L,4L),(U,0),L;U,ν=U*L/Re,body,ϵ,T=Float64)

t₀ = round(sim_time(sim))
duration = 10
tstep = 0.1
ωᵣ = 0.5

# force function
integration_points = Splines.uv_integration(p)

# intialise coupling
f_old = force(body,sim); size_f = size(f_old)
pnts_old = zero(u⁰); pnts_old .+= u⁰

# set up coupling
QNCouple = Relaxation(reshape(dⁿ[1:2p.mesh.numBasis],(p.mesh.numBasis,2))',0.0.*f_old;relax=ωᵣ)
# QNCouple = IQNCoupling(reshape(dⁿ[1:2p.mesh.numBasis],(p.mesh.numBasis,2))',0.0.*f_old;relax=ωᵣ)
updated_values = zero(QNCouple.x)

# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    
    global dⁿ, vⁿ, aⁿ, updated_values, f_old, pnts_old;

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])

    while t < tᵢ*sim.L/sim.U

        # save at start of iterations
        WaterLily.store!(sim.flow)
        cache = (dⁿ, vⁿ, aⁿ)
        
        # time steps
        Δt = sim.flow.Δt[end]/sim.L*sim.U
        tⁿ = t/sim.L*sim.U; # previous time instant
        tⁿ⁺¹ = tⁿ + Δt; # current time instal

        # implicit solve
        iter = 1
        while true

            dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹ = Splines.step2(jacob, stiff, Matrix(M), resid, fext, f_old,
                                             dⁿ, vⁿ, aⁿ, tⁿ, tⁿ⁺¹, αm, αf, β, γ, p)
            pnts_new = u⁰+reshape(L*dⁿ⁺¹[1:2mesh.numBasis],(mesh.numBasis,2))'
            
            # update the body
            ParametricBodies.update!(body,pnts_old,sim.flow.Δt[end])
            
            # update the flow
            measure!(sim,t); mom_step!(sim.flow,sim.pois)

            # compute forces
            f_new = force(body,sim)
            # step the structure in time
            if tⁿ<=2.0
                f_new[2,:] .+= 25*t
            end

            # check that residuals have converged
            rd = res(pnts_old,pnts_new); rf = res(f_old,f_new);
            println("    Iter: ",iter,", rd: ",round(rd,digits=8),", rf: ",round(rf,digits=8))
            if ((rd<1e-2) && (rf<1e-2)) || iter+1 > 50 # if we converge, we exit to avoid reverting the flow
                println("  Converged...")
                dⁿ, vⁿ, aⁿ = dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹
                f_old .= f_new; pnts_old .= pnts_new
                break
            end

            # accelerate coupling
            concatenate!(updated_values, reshape(L*dⁿ⁺¹[1:2p.mesh.numBasis],(p.mesh.numBasis,2))', f_new, QNCouple.subs)
            updated_values = update(QNCouple, updated_values, iter==1)
            revert!(updated_values, pnts_old, f_old, QNCouple.subs)
            pnts_old .= u⁰ .+ pnts_old

            # if we have not converged, we must revert
            WaterLily.revert!(sim.flow)
            dⁿ, vⁿ, aⁿ = cache
            iter += 1

        end

        # finalize
        t += sim.flow.Δt[end]
    end
    
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim);plot_vorticity(sim.flow.σ, limit=10)
    plot!(body.surf)
end
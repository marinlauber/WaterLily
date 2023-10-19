using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
using SparseArrays
include("examples/TwoD_plots.jl")

# relative resudials
res(a,b) = norm(a-b)/norm(b)

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
A = 0.1
Ii = 1e-3
E = 1000.0
L = 1.0
EI = E*Ii #1.0
EA = E*A #10.0
f(s) = [0.0,0.0] # s is curvilinear coordinate
P = 3EI/2

# natural frequencies
ωₙ = 1.875; fhz = 0.125
density(ξ) = 0.0*(ωₙ^2/2π)^2*(EI/(fhz^2*L^4))
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
fext = zeros(size(resid)); loading = zeros(size(resid))
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
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[1.5L,2L])
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
Body = DynamicBody(nurbs, (0,1));
sim = Simulation((4L,6L),(0,U),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)

t₀ = round(sim_time(sim))
duration = 1
tstep = 0.1

# force functions
integration_points = Splines.uv_integration(p)
f_old = zeros((2,length(integration_points)))
pnts_old = copy(u⁰)

# time loop
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)
    
    global dⁿ, vⁿ, aⁿ,f_old,pnts_old;

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])

    while t < tᵢ*sim.L/sim.U

        println("tᵢ=$tᵢ, t=$t, Δt=$(sim.flow.Δt[end])")
        
        # save the state
        WaterLily.store!(sim.flow)
        
        # implicit solve
        iter = 1; r₂ = 1.0;
        # f_new=f_old; pnts_new=pnts_old;
        
        # time steps
        Δt = sim.flow.Δt[end]/sim.L*sim.U
        tⁿ = t/sim.L*sim.U; # previous time instant
        tⁿ⁺¹ = tⁿ + Δt; # current time instal

        while r₂ < 1e-3
            println("residuals: ",rand()/iter^2)

            # ωᵣ = 0.01
            # # set initial values and relax
            # pnts_old = u⁰+reshape(L*dⁿ[1:2p.mesh.numBasis],(p.mesh.numBasis,2))'
            # pnts_old = (1-ωᵣ)*pnts_old .+ ωᵣ*pnts_new
            # # f_old = NurbsForce(body.surf,sim.flow.p,integration_pnts)
            # f_old .= 0.0
            # f_old[2,:] .= 2P*sin(2π*fhz*tⁿ)
            # f_old = (1-ωᵣ)*f_old .+ ωᵣ*f_new
            
            
            # step the structure in time
            # dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹ = Splines.step(jacob, stiff, Matrix(M), resid, fext, loading, dⁿ, vⁿ, aⁿ, tⁿ, tⁿ⁺¹, αm, αf, β, γ, p)
            # dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹ = Splines.step2(jacob, stiff, Matrix(M), resid, fext, f_old, dⁿ, vⁿ, aⁿ, tⁿ, tⁿ⁺¹, αm, αf, β, γ, p)
            # pnts_new = u⁰+reshape(L*dⁿ⁺¹[1:2p.mesh.numBasis],(p.mesh.numBasis,2))'
            
            # update the body and the flow
            # ParametricBodies.update!(Body,pnts_old,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            # f_new = reduce(hcat,[ParametricBodies.NurbsForce(Body.surf,sim.flow.p,s) for s ∈ integration_points])

            # # check that residuals have converged
            # rd=res(pnts_new,pnts_old); rf=res(f_new,f_old);
            # println("iter=",iter,", rd=",round(rd,digits=3),", rf=",round(rf,digits=3))

            # if ((rd<1e-2) && (rf<1e-2))
            if iter>3 # if we converge, we exit to avoid reverting the flow
                println("Converged...moving to next time step")
                break
            end

            # if not converged, revert to last state
            WaterLily.revert!(sim.flow)
            iter += 1
        end

        # finalize
        t += sim.flow.Δt[end]
    end
    
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    plot!(Body.surf)
end
gif(anim, "perpendicular_flap.gif"; fps=20)

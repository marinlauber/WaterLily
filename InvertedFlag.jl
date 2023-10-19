using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
using SparseArrays
include("examples/TwoD_plots.jl")

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
A = 0.1
Ii = 1e-6
E = 10000000.0
L = 1.0
EI = E*Ii #1.0
EA = E*A #10.0
f(s) = [0.0,0.0] # s is curvilinear coordinate
P = 3EI/2

# natural frequencies
ωₙ = 1.875; fhz = 0.125
density(ξ) = (ωₙ^2/2π)^2*(EI/(fhz^2*L^4))
println(ωₙ^2.0*√(EI/(density(0.5)*L^4))/(2π))

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
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[L,2L])
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
Body = DynamicBody(nurbs, (0,1));
sim = Simulation((6L,4L),(U,0),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)

t₀ = round(sim_time(sim))
duration = 10
tstep = 0.1

# force functions
integration_points = Splines.uv_integration(p)
loading = zeros((2,length(integration_points)))

fort9 = open("test.txt","w")

# time loop
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)
    
    global dⁿ, vⁿ, aⁿ;

    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        
        # time steps
        Δt = sim.flow.Δt[end]/sim.L*sim.U
        tⁿ = t/sim.L*sim.U; # previous time instant
        tⁿ⁺¹ = tⁿ + Δt; # current time instal

        # step the structure in time
        # loading[p.mesh.numBasis+1] = P*sin(2π*fhz[1]*tⁿ)
        loading .= 0.0
        loading[2,:] .= 2P*sin(2π*fhz*tⁿ)
        dⁿ, vⁿ, aⁿ = Splines.step2(jacob, stiff, Matrix(M), resid, fext, loading, dⁿ, vⁿ, aⁿ, tⁿ, tⁿ⁺¹, αm, αf, β, γ, p)
        loading .= reduce(hcat,[ParametricBodies.NurbsForce(Body.surf,sim.flow.p,s) for s ∈ integration_points])
        println(fort9,round.(loading,digits=3))
        
        # extract solution and update geometry 
        new_pnts = u⁰+reshape(L*dⁿ[1:2mesh.numBasis],(mesh.numBasis,2))'
        # println(new_pnts)
        ParametricBodies.update!(Body,new_pnts,sim.flow.Δt[end])

        # update the body
        measure!(sim,t)

        # update the flow
        mom_step!(sim.flow,sim.pois)

        # finalize
        t += sim.flow.Δt[end]
    end
    
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    # fₓ = ParametricBodies.force(Body.surf,sim.flow.p); println(fₓ/L)
    get_omega!(sim);plot_vorticity(sim.flow.σ, limit=10)
    plot!(Body.surf)
end
close(fort9)
gif(anim, "inverted_flag.gif"; fps=20)
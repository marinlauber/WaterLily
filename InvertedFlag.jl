# using WaterLily
# using ParametricBodie
using StaticArrays
using LinearAlgebra
using SparseArrays
using Plots

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(clamp.(ω,-limit,limit)',dpi=300,
    color=palette(:RdBu_11), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)


function NurbsForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T
    xᵢ = surf(s,0.0)
    δnᵢ = δ*ParametricBodies.norm_dir(surf,s,0.0); δnᵢ/=√(δnᵢ'*δnᵢ)
    Δpₓ =  WaterLily.interp(xᵢ+δnᵢ,p)
    Δpₓ -= WaterLily.interp(xᵢ-δnᵢ,p)
    return -Δpₓ.*δnᵢ
end

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
A = 0.1
I = 1e-3
E = 1000.0
L = 1.0
EI = E*I #1.0
EA = E*A #10.0
f(s) = [0.0,0.0] # s is curvilinear coordinate

density(ξ) = A*0.0
P = 3EI/2
exact_sol(x) = P.*x.^2/(6EI).*(3 .- x) # fixed - free (Ponts Load)

# natural frequencies
ωₙ = [1.875, 4.694, 7.855]
fhz = ωₙ.^2.0.*√(EI/(density(0.5)*L^4))/(2π)
display(fhz)
fhz = [0.25,2,3]

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

# time steps
Δt = 0.1
T = 5.0/fhz[1]
time = collect(0.0:Δt:T);
Nₜ = length(time);

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

# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

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

tᵢ = round(sim_time(sim))
duration = 10

# force functions
# f(s) = NurbsForce(Body.surf,sim.flow.p,s)

# time loop
anim = @animate while tᵢ ≤ duration*sim.L/sim.U

    global tᵢ,dⁿ, vⁿ, aⁿ, F;
    global vⁿ⁺¹, aⁿ⁺¹, dⁿ⁺¹, dⁿ⁺ᵅ, vⁿ⁺ᵅ, aⁿ⁺ᵅ;

    # update until time tᵢ in the background
    measure!(sim,tᵢ)
    mom_step!(sim.flow,sim.pois)
            
    # structural time steps
    Δt = sim.flow.Δt[end]/sim.L*sim.U
    tⁿ = WaterLily.time(sim)/sim.L*sim.U; # previous time instant
    tⁿ⁺¹ = tⁿ+Δt; # current time instal
    tⁿ⁺ᵅ = αf*tⁿ⁺¹ + (1.0-αf)*tⁿ;

    # predictor (initial guess) for the Newton-Raphson scheme
    # d_{n+1}
    dⁿ⁺¹ = dⁿ; r₂ = 1.0; iter = 1;

    # Newton-Raphson iterations loop
    while r₂ > 1.0e-6 && iter < 100
        # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
        vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1.0-γ/β)*vⁿ - Δt*(γ/2β-1.0)*aⁿ;
        aⁿ⁺¹ = 1.0/(β*Δt^2)*dⁿ⁺¹ - 1.0/(β*Δt^2)*dⁿ - 1.0/(β*Δt)*vⁿ - (1.0/2β-1.0)*aⁿ;

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = αf*dⁿ⁺¹ + (1.0-αf)*dⁿ;
        vⁿ⁺ᵅ = αf*vⁿ⁺¹ + (1.0-αf)*vⁿ;
        aⁿ⁺ᵅ = αm*aⁿ⁺¹ + (1.0-αm)*aⁿ;
    
        # update stiffness and jacobian, linearised here
        Splines.update_global!(stiff, jacob, dⁿ⁺ᵅ, p.mesh, p.gauss_rule, p)
    
        # update rhs vector
        Splines.update_external!(fext, p.mesh, p.f, p.gauss_rule)
        fext[mesh.numBasis+1] += P*sin(2π*fhz[1]*tⁿ⁺ᵅ);

        # # apply BCs
        jacob .= αm/(β*Δt^2)*M + αf*jacob
        resid .= stiff*dⁿ⁺ᵅ + M*aⁿ⁺ᵅ - fext
        Splines.applyBCGlobal!(stiff, jacob, resid, p.mesh, 
                               p.Dirichlet_BC, p.Neumann_BC,
                               p.gauss_rule)

        # check convergence
        r₂ = norm(resid);
        if r₂ < 1.0e-6 && break; end

        # newton solve for the displacement increment
        dⁿ⁺¹ -= jacob\resid; iter += 1
    end
    
    # copy variables ()_{n} <-- ()_{n+1}
    dⁿ = dⁿ⁺¹;
    vⁿ = vⁿ⁺¹;
    aⁿ = aⁿ⁺¹;

    # extract solution and update geometry 
    Body.surf.pnts .= u⁰+reshape(L*dⁿ[1:2mesh.numBasis],(mesh.numBasis,2))'
    ParametricBodies.update!(Body,tᵢ)

    # plot stuff
    if mod(tᵢ, 0.1*sim.L/sim.U) < sim.flow.Δt[end]
        println("tU/L=",round(tᵢ*sim.U/sim.L,digits=4),
                ", Δt=",round(sim.flow.Δt[end],digits=3))
        force = sum(reduce(hcat, [NurbsForce(Body.surf,sim.flow.p,s) for s=0:0.01:1]), dims=2)
        println(force/L)
        ti =round(tᵢ/sim.L*sim.U,digits=3)
        get_omega!(sim);
        plot_vorticity(sim.flow.σ, limit=10)
        Plots.plot!(Body.surf.pnts[1,:],Body.surf.pnts[2,:],markers=:o,legend=false,
                    aspect_ratio=:equal, title="t = $ti")
        Xs = reduce(hcat,[Body.surf(s,0.0) for s ∈ 0:0.01:1])
        Plots.plot!(Xs[1,:],Xs[2,:],color=:black,lw=thk,legend=false)
    end

    # finish time step
    tᵢ += sim.flow.Δt[end]
end
gif(anim, "inverted_flag.gif"; fps=200)

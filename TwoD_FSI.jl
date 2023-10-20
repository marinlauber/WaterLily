using WaterLily
using ParametricBodies
using Splines
using LinearAlgebra
using StaticArrays
include("examples/TwoD_plots.jl")

function force(b::DynamicBody,sim::Simulation)
    reduce(hcat,[ParametricBodies.NurbsForce(b.surf,sim.flow.p,s) for s ∈ integration_points])
end

abstract type AbstractCoupling end

struct Relaxation <: AbstractCoupling
    ω :: Float64
    x :: AbstractArray{Float64}
    x_s :: AbstractArray{Float64}
    function Relaxation(x⁰::AbstractArray{Float64},xs⁰::AbstractArray{Float64};ω::Float64=0.5)
        new(0.5,zero(x⁰),zero(xs⁰))
    end
end
function update(cp::Relaxation, x_new, x_new_s) 
    # relax primary data
    r = x_new .- cp.x
    cp.x .= x_new
    x_new .+= cp.ω.*r
    # relax secondary data
    r_s = x_new_s .- cp.x_s
    cp.x_s .= x_new_s
    x_new_s .+= cp.ω.*r_s
    return x_new, x_new_s
end

struct IQNCoupling <: AbstractCoupling
    ω :: Float64
    x :: AbstractArray{Float64}
    r :: AbstractArray{Float64}
    x_s :: AbstractArray{Float64}
    r_s :: AbstractArray{Float64}
    V :: AbstractArray{Float64}
    W :: AbstractArray{Float64}
    Ws :: AbstractArray{Float64}
    iter :: Vector{Int64}
    function IQNCoupling(x⁰::AbstractArray{Float64},xs⁰::AbstractArray{Float64};ω::Float64=0.5)
        N1=length(x⁰); N2=length(xs⁰)
        # Ws is of size (N2,N1) as there are only N1 cᵏ
        new(ω,zero(x⁰),zero(x⁰),zero(xs⁰),zero(xs⁰),zeros(N1,N1),zeros(N1,N1),zeros(N2,N1),[0])
    end
end
function backsub(A,b)
    n = size(A,1)
    x = zeros(n)
    x[n] = b[n]/A[n,n]
    for i in n-1:-1:1
        s = sum( A[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / A[i,i]
    end
    return x
end
function update(cp::IQNCoupling, x_new, x_new_s)
    if cp.iter[1]==0 # relaxation step
        # relax primary data
        r = x_new .- cp.x
        cp.x .= x_new
        cp.r .= r
        x_new .+= cp.ω.*r
        # relax secondary data
        r_s = x_new_s .- cp.x_s
        cp.x_s .= x_new_s
        cp.r_s .= r_s
        x_new_s .+= cp.ω.*r_s
        # cp.iter[1] = 1
    else
        k = cp.iter[1]; N = length(cp.x)
        # compute residuals
        r  = x_new .- cp.x
        r_s= x_new_s .- cp.x_s
        # roll the matrix to make space for new column
        roll!(cp.V); roll!(cp.W); roll!(cp.Ws)
        cp.V[:,1] = r .- cp.r; cp.r .= r
        cp.W[:,1] = x_new .- cp.x; cp.x .= x_new
        cp.Ws[:,1] = x_new_s .- cp.x_s; cp.x_s .= x_new_s # secondary data
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view cp.V[:,1:min(k,N)])
        cᵏ = backsub(Rᵏ,-Qᵏ'*r)
        x_new   .+= (@view cp.W[:,1:min(k,N)])*cᵏ #.+ rᵏ #not sure
        x_new_s .+= (@view cp.Ws[:,1:min(k,N)])*cᵏ # secondary data
        cp.iter[1] = k + 1
    end
    return x_new, x_new_s
end
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])

# relative resudials
res(a,b) = norm(a-b)/norm(b)

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
ωₙ = 1.875; fhz = 0.0125
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
L=2^4
Re=500
U=1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# construct from mesh, this can be tidy
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:]*L.+[3L,2L].+1.5)
nurbs = NurbsCurve(copy(u⁰),mesh.knots,mesh.weights)

# flow sim
body = DynamicBody(nurbs, (0,1));

# make a simulation
sim = Simulation((4L,8L), (0,U), L; ν=U*L/Re, body, T=Float64)

# duration of the simulation
duration = 20
step = 0.1
t₀ = 0.0
ωᵣ = 0.25 # ωᵣ ∈ [0,1] is the relaxation parameter

# force functions
integration_points = Splines.uv_integration(p)

# intialise coupling
# f_old = zeros((2,length(integration_points))); f_old[2,:] .= 2P*sin(2π*fhz*0.0)
f_old = force(body,sim); size_f = size(f_old)
pnts_old = zero(u⁰); pnts_old .+= u⁰

# coupling
relax = IQNCoupling([pnts_old...],[f_old...];ω=ωᵣ)
# relax = Relaxation([pnts_old...],[f_old...];ω=ωᵣ)

# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step)

    global dⁿ, vⁿ, aⁿ, f_old, pnts_old;

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])

    while t < tᵢ*sim.L/sim.U
        
        println("  tᵢ=$tᵢ, t=$(round(t,digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        WaterLily.store!(sim.flow)
        cache = (dⁿ, vⁿ, aⁿ)
        
        # time steps
        Δt = sim.flow.Δt[end]/sim.L*sim.U
        tⁿ = t/sim.L*sim.U; # previous time instant
        tⁿ⁺¹ = tⁿ + Δt; # current time install
        
        # implicit solve
        iter = 1; r₂ = 1.0;

        # iterative loop
        while true
            
            # update the structure
            dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹ = Splines.step2(jacob, stiff, Matrix(M), resid, fext, f_old, dⁿ, vⁿ, aⁿ, tⁿ, tⁿ⁺¹, αm, αf, β, γ, p)
            pnts_new = u⁰+reshape(L*dⁿ⁺¹[1:2p.mesh.numBasis],(p.mesh.numBasis,2))'
            
            # update flow
            ParametricBodies.update!(body,pnts_old,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            f_new = min(1.0,5tᵢ).*force(body,sim)
            # f_new = zero(f_old); f_new[2,:] .= 2P*sin(2π*fhz*tⁿ⁺¹)

            # check that residuals have converged
            rd = res(pnts_new,pnts_old); rf = res(f_new,f_old);
            println("    Iter: ",iter,", rd: ",round(rd,digits=8),", rf: ",round(rf,digits=8))
            if ((rd<1e-3) && (rf<1e-3)) || iter > 20 # if we converge, we exit to avoid reverting the flow
                println("  Converged...")
                dⁿ, vⁿ, aⁿ = dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹
                break
            end

            # accelerate coupling
            # pnts_old, f_old = accelerate(pnts_old, f_old, pnts_new, f_new, ωᵣ)
            pnts_old, f_old = update(relax, [pnts_new...], [f_new...])
            # f_old, pnts_old = update(relax, [f_new...], [pnts_new...])
            pnts_old = reshape(pnts_old, (2,p.mesh.numBasis))
            f_old    = reshape(f_old, size_f)

            # if we have not converged, we must revert
            WaterLily.revert!(sim.flow)
            dⁿ, vⁿ, aⁿ = cache
            iter += 1
        end

        # finish the time step
        Δt = sim.flow.Δt[end]
        t += Δt
    end

    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    plot!(body.surf, show_cp=false)
    plot!(title="tU/L $tᵢ")
    
end

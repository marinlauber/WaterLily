using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
include("examples/TwoD_plots.jl")

function force(b::DynamicBody,sim::Simulation)
    reduce(hcat,[ParametricBodies.NurbsForce(b.surf,sim.flow.p,s) for s ∈ integration_points])
end

struct IQNCoupling2 <: WaterLily.AbstractCoupling
    ω :: Float64                    # intial relaxation
    x :: AbstractArray{Float64}     # primary variable
    x̃ :: AbstractArray{Float64}     # old solver iter (not relaxed)
    r :: AbstractArray{Float64}     # primary residual
    V :: AbstractArray{Float64}     # primary residual difference
    W :: AbstractArray{Float64}     # primary variable difference
    subs                            # sub residual indices
    iter :: Dict{Symbol,Int64}      # iteration counter
    function IQNCoupling2(primary::AbstractArray{Float64},secondary::AbstractArray;relax::Float64=0.5)
        n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂
        subs = (1:m₁,m₁+1:n₁*m₁,n₁*m₁+1:n₁*m₁+m₂,n₁*m₁+m₂+1:N)
        x⁰ = zeros(N); concatenate!(x⁰,primary,secondary,subs)
        new(relax,x⁰,zeros(N),zeros(N),zeros(N,N),zeros(N,N),subs,Dict(:k => 0))
    end
end
function concatenate!(vec, a, b, subs)
    vec[subs[1]] = a[1,:];
    vec[subs[2]] = a[2,:];
    vec[subs[3]] = b[1,:];
    vec[subs[4]] = b[2,:];
end
function revert!(vec, a, b, subs)
    a[1,:] = vec[subs[1]];
    a[2,:] = vec[subs[2]];
    b[1,:] = vec[subs[3]];
    b[2,:] = vec[subs[4]];
end
function preconditonner(r,subs,reset)
    Φᵏ = ones(length(r))
    reset && return Diagonal(Φᵏ);
    for s in subs
        Φᵏ[s] .= norm(r)/norm(r[s])
    end
    Diagonal(Φᵏ)
end
function Q1filter!(A;ϵ=1e-8)
    N,_ = size(A); normA=norm(A)
    for i ∈ N:-1:1
        A[i,i]<ϵ*normA && popCol!(A,i);
    end
end
function updateQN2(cp::IQNCoupling2, xᵏ)
    if cp.iter[:k]==0
        # store variable and residual
        rᵏ = xᵏ .- cp.x; cp.x̃.=xᵏ
        # relaxation update
        xᵏ .= cp.x .+ cp.ω*rᵏ
        # store
        cp.x.=xᵏ; cp.r.=rᵏ
    else
        # residuals
        N = length(cp.x);
        rᵏ = xᵏ .- 
        # roll the matrix to make space for new column
        WaterLily.roll!(cp.V); WaterLily.roll!(cp.W)
        cp.V[:,1] = rᵏ .- cp.r; cp.r .= rᵏ
        cp.W[:,1] = xᵏ .- cp.x̃; cp.x̃ .= xᵏ # save old solver iter
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view cp.V[:,1:min(cp.iter[:k],N)])
        cᵏ = WaterLily.backsub(Rᵏ,-Qᵏ'*rᵏ)
        xᵏ.= cp.x .+ (@view cp.W[:,1:min(cp.iter[:k],N)])*cᵏ .+ rᵏ #not sure
        # update for next step
        cp.x .= xᵏ
    end
    # cp.iter[:k] += 1
    return xᵏ
end
function updateQN(cp::IQNCoupling2, x_new)
    if cp.iter[:k]==0 # relaxation step
        # relax primary data
        r = x_new .- cp.x
        # store new values
        cp.x .= x_new; cp.r .= r
        x_new .+= cp.ω.*r
    else
        k = cp.iter[:k]; N = length(cp.x)
        # compute residuals
        r = x_new .- cp.x
        # roll the matrix to make space for new column
        WaterLily.roll!(cp.V); WaterLily.roll!(cp.W)
        cp.V[:,1] = r .- cp.r;      cp.r .= r
        cp.W[:,1] = x_new .- cp.x;  cp.x .= x_new
        # preocndition system
        Φᵏ=preconditonner(r,cp.subs,true); Vᵏ = Φᵏ*cp.V
        # Q1filter!(Vᵏ; ϵ=1e-8)
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view Vᵏ[:,1:min(k,N)])
        cᵏ = WaterLily.backsub(Rᵏ,-Qᵏ'*Φᵏ*r)
        x_new .+= (@view cp.W[:,1:min(k,N)])*cᵏ # .+ r # not sure
    end
    # cp.iter[:k] += 1
    return x_new
end
popCol!(A::AbstractArray,k) = (A[:,k:end-1] .= A[:,k+1:end]; A[:,end].=0)


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
duration = 10.0
step = 0.1
t₀ = 0.0
ωᵣ = 0.25 # ωᵣ ∈ [0,1] is the relaxation parameter

# force functions
integration_points = Splines.uv_integration(p)

# intialise coupling
f_old = force(body,sim); size_f = size(f_old)
pnts_old = zero(u⁰); pnts_old .+= u⁰

# coupling
# relax = IQNCoupling([f_old...],[pnts_old...];relax=ωᵣ)
# relax = Relaxation([pnts_old...],[f_old...];relax=ωᵣ)

QNCouple = IQNCoupling2(reshape(dⁿ[1:2p.mesh.numBasis],(p.mesh.numBasis,2))',0.0.*f_old;relax=ωᵣ)
updated_values = zero(QNCouple.x)

# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step)

    global dⁿ, vⁿ, aⁿ, f_old, pnts_old, updated_values;

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
        tⁿ⁺¹ = tⁿ + Δt;     # current time install
        
        # implicit solve
        iter=1; new=true

        # iterative loop
        while true
            
            # update the structure
            dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹ = Splines.step2(jacob, stiff, Matrix(M), resid, fext, f_old, dⁿ, vⁿ, aⁿ, tⁿ, tⁿ⁺¹, αm, αf, β, γ, p)
            pnts_new = u⁰+reshape(L*dⁿ⁺¹[1:2p.mesh.numBasis],(p.mesh.numBasis,2))'
            
            # update flow
            ParametricBodies.update!(body,pnts_old,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            f_new = force(body,sim)

            # check that residuals have converged
            rd = res(pnts_old,pnts_new); rf = res(f_old,f_new);
            println("    Iter: ",iter,", rd: ",round(norm(pnts_old-pnts_new),digits=8),", rf: ",round(rf,digits=8))
            if ((rd<1e-2) && (rf<1e-2)) || iter > 16 # if we converge, we exit to avoid reverting the flow
                println("  Converged...")
                dⁿ, vⁿ, aⁿ = dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹
                break
            end

            # accelerate coupling
            concatenate!(updated_values, reshape(L*dⁿ⁺¹[1:2p.mesh.numBasis],(p.mesh.numBasis,2))', f_new, QNCouple.subs)
            updated_values = updateQN(QNCouple, updated_values)
            revert!(updated_values, pnts_old, f_old, QNCouple.subs)
            pnts_old .= u⁰ .+ pnts_old

            # pnts_old, f_old = accelerate(pnts_old, f_old, pnts_new, f_new, ωᵣ)
            # pnts_old, f_old = update(relax, [pnts_new...], [f_new...])
            # f_old, pnts_old = update(relax, [f_new...], [pnts_new...])
            # pnts_old = reshape(pnts_old, (2,p.mesh.numBasis))
            # f_old    = reshape(f_old, size_f)
           
            # if we have not converged, we must revert
            WaterLily.revert!(sim.flow)
            dⁿ, vⁿ, aⁿ = cache
            iter += 1
            new = false
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

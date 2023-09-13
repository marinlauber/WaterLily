using WaterLily
using Plots
using StaticArrays
using LinearAlgebra
using Pkg
# Pkg.activate("/home/marin/Workspace/ParametricBodies.jl")
# using ParametricBodies
# Pkg.activate("/home/marin/Workspace/Splines.jl")
# using Splines

function rotate_(vec, alpha::T, axis) where {T}
    cs=cos(alpha); ss=sin(alpha)
    R = Matrix{T}(I,3,3)
    d1=axis%3+1; d2=(axis+1)%3+1
    R[d1,d1]=cs; R[d1,d2]=-ss
    R[d2,d1]=ss; R[d2,d2]=cs
    return R*vec
end
function rotate(a, alpha, k)
    b = copy(a)
    for i in 1:size(a,2)
        b[:,i] = rotate_(a[:,i], alpha, k)
    end
    return b
end

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(clamp.(ω,-limit,limit)',dpi=300,
    color=palette(:RdBu_11), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)

# # fluid force at x
# function p_force(p::AbstractArray{T},body::ParametricBody,gᵢ,δ=2.0) where T
#     xᵢ = body.surf(gᵢ,0.0)
#     δnᵢ = δ*ParametricBodies.norm_dir(body.surf,gᵢ,0.0); δnᵢ/=WaterLily.norm2(δnᵢ)
#     Δpₓ =  WaterLily.interp(convert.(T,xᵢ+δnᵢ),p)
#     Δpₓ -= WaterLily.interp(convert.(T,xᵢ-δnᵢ),p)
#     return -Δpₓ.*δnᵢ
# end

function NurbsForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T
    xᵢ = surf(s,0.0)
    δnᵢ = δ*ParametricBodies.norm_dir(surf,s,0.0); δnᵢ/=√(δnᵢ'*δnᵢ)
    Δpₓ =  WaterLily.interp(xᵢ+δnᵢ,p)
    Δpₓ -= WaterLily.interp(xᵢ-δnᵢ,p)
    return -Δpₓ.*δnᵢ
end

# FEA constructor
deg=3; numElem=2
mesh, gauss_rule = Splines.Mesh1D(0,1,numElem,deg)

# Material properties and mesh
EI = 1.0
EA = 10.0

# boundary conditions
ptLeft = 0.0
ptRight = 1.0
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1)
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2)
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1)
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]

## Simulation parameters
L=2^5
Re=2500
U=1
ϵ=0.5
thk=2ϵ+√2

# force functions
f_rhs(phys_pt) = NurbsForce(Body.surf,sim.flow.p,phys_pt)

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# make a problem
p = EulerBeam(EI, EA, f_rhs, mesh, gauss_rule, Dirichlet_BC)

# construct from mesh, this can be tidy
pts_rotated = rotate(mesh.controlPoints, -π/16, 3)[1:2,:]
u⁰ = MMatrix{2,size(mesh.controlPoints,2)}(pts_rotated*L.+[L,2L])
nurbs = NurbsCurve(u⁰,mesh.knots,mesh.weights)

# flow sim
Body = ParametricBody(nurbs, (0,1));
sim = Simulation((6L,4L),(U,0),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)

t₀ = round(sim_time(sim))
duration = 20; tstep = 0.1; force = []
ω = 0.5 # relaxation parameters
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)

    # update structure, this computes the forces
    result = static_lsolve!(p, static_residuals!, static_jacobian!)

    # extract solution and update geometry
    uⁿ = copy(Body.surf.pnts)
    uⁿ⁺¹ = u⁰+reshape(L*result[1:2mesh.numBasis],(mesh.numBasis,2))'
    Body.surf.pnts .= uⁿ + ω*(uⁿ⁺¹ - uⁿ)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
    Plots.plot!(Body.surf.pnts[1,:],Body.surf.pnts[2,:],markers=:o,legend=false, border=:none)
    Xs = reduce(hcat,[Body.surf(s,0.0) for s ∈ 0:0.01:1])
    Plots.plot!(Xs[1,:],Xs[2,:],color=:black,lw=thk,legend=false, border=:none)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# forces = reduce(hcat,force)

# result = static_lsolve!(p, static_residuals!, static_jacobian!)

# # extract solution
# u0 = result[1:mesh.numBasis]
# w0 = result[mesh.numBasis+1:2mesh.numBasis]
# u = getSol(mesh, u0, 100)
# x = LinRange(ptLeft, ptRight, length(u))
# u += x
# w = getSol(mesh, w0, 100)
# Plots.plot(u, w, label="Sol")
# Plots.plot!(mesh.controlPoints[1,:] .+ u0, w0, marker=:o, label="Control points")

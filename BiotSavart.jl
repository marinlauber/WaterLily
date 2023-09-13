
using WaterLily
using LinearAlgebra
include("examples/TwoD_plots.jl")
"""
Biot-Savart integral

Compute the biot-savart for all the components of the velocity at every point in the domain
"""
function BiotSavart!(u,ω)
    N,n = WaterLily.size_u(u)
    u .= 0.0
    for i ∈ 1:n
        j=i%2+1 # the only component not zero in the vorticity
        for I ∈ WaterLily.inside_u(N,i)
            @WaterLily.loop u[I,i] += K(i,I,j,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
        end
    end
end
function Integrate!(ψ,ω)
    ψ .= 0.0
    for I ∈ CartesianIndices(ψ)
        @WaterLily.loop ψ[I] += G∞(I,J)*ω[J] over J ∈ CartesianIndices(ω)
    end
end
"""
 Laplacian kernel
"""
function G∞(I,J) # ψ and ω is on the edge of the cell
    I==J && return 0.0 # ignore self contribution
    return log(norm(loc(0,I).-loc(0,J)))/2π
end
"""
Biot-Savart kernel
"""
function K(i,I,j,J,ϵ=1e-6)
    # face centered velocity at I due to vorticity at cell edge J
    r = loc(i,I) .- loc(0,J) .+ 0.5 # cell edge vorticity
    return sign(i-j)*r[j]/(2π*norm(r)^2+ϵ^2)
end

"""
RankineVortex(i,xy,center,R,Γ)
"""
function RankineVortex(i, xy, center, R=4, Γ=1)
    xy = (xy .- 1.5 .- center)
    x,y = xy
    θ = atan(y,x)
    r = norm(xy)
    vθ =Γ/2π*(r<=R ? r/R^2 : 1/r)
    v = [-vθ*sin(θ),vθ*cos(θ)]
    return v[i]
end
function SourceSink(i,xy,center;Γ=1.0)
    xy = (xy .- 1.5 .- center); x,y = xy
    θ,r = atan(y,x),norm(xy)
    v = Γ/(2π*r).*[cos(θ),sin(θ)]
    return v[i]
end
function Vortex(i,xy,center;Γ=1.0)
    xy = (xy .- 1.5 .- center); x,y = xy
    θ,r = atan(y,x),norm(xy)
    v = Γ/(2π*r).*[-sin(θ),cos(θ)]
    return v[i]
end
function Helmoltz!(ϕ,ψ,a)
    # compute vorticity
    @inside a.σ[I] =-WaterLily.curl(3,I,a.u)
    # compute streamfunction from vorticity
    Integrate!(ψ,a.σ);
    # compute divergence
    @inside a.σ[I] = WaterLily.div(I,a.u)
    # compute potential from divergence
    Integrate!(ϕ,a.σ);
end
function Helmoltz!(uϕ,uψ,ϕ,ψ,a::Flow{n}) where n
    Helmoltz!(ϕ,ψ,a)
    for i ∈ 1:n
        j = i%2+1 # other component
        # compute gradient of potential
        # @WaterLily.loop uϕ[I,i] = WaterLily.∂(i,I,ϕ) over I ∈ inside(ϕ)
        # compute velocity from stream function, ψ is edge-centered
        @WaterLily.loop uψ[I,i] = sign(j-i)*WaterLily.∂(j,I+δ(j,I),ψ) over I ∈ inside(ψ)
    end
    # uψ .= sim.flow.u .- uϕ
    uϕ .= sim.flow.u .- uψ
end
# function Helmoltz!(ϕ,ψ,a,b)
#     # compute vorticity
#     @inside b.z[I] =-WaterLily.curl(3,I,a.u)
#     # compute streamfunction from vorticity
#     solver!(b); BC!(b.x); ψ .= b.x
#     # compute divergence
#     @inside b.z[I] = WaterLily.div(I,a.u)
#     # compute potential from divergence
#     solver!(b); BC!(b.x); ϕ .= b.x
# end
# function Helmoltz!(uϕ,uψ,ϕ,ψ,a::Flow{n},pois) where n
#     Helmoltz!(ϕ,ψ,a,pois)
#     for i ∈ 1:n
#         j = i%2+1 # other component
#         # compute gradient of potential
#         @WaterLily.loop uϕ[I,i] = WaterLily.∂(i,I,ϕ) over I ∈ inside(ϕ)
#         # compute velocity from stream function, ψ is edge-centered
#         @WaterLily.loop uψ[I,i] = sign(j-i)*WaterLily.∂(j,I+δ(j,I),ψ) over I ∈ inside(ψ)
#     end
#     # uψ .= sim.flow.u .- uϕ
# end
# some definitons
U = 1
Re = 250
m, n = 2^5, 2^5

# make a simulation
sim = Simulation((n,m), (U,0), m; ν=U*m/Re, T=Float64)
ψ = copy(sim.flow.p); ϕ = copy(sim.flow.p); p = copy(sim.flow.p)
uϕ,uψ=copy(sim.flow.u),copy(sim.flow.u)
# pois = MultiLevelPoisson(p,sim.flow.μ₀,sim.flow.σ)

# make a Rankine vortex
# f(i,x) = RankineVortex(i,x,(m/2,m/2),10, 1)
f(i,x) = SourceSink(i,x,(m/4,m/4),Γ=1.0).+SourceSink(i,x,(3m/4,3m/4),Γ=-1.0).+
         Vortex(i,x,(3m/4,m/4),Γ=1.0).+Vortex(i,x,(m/4,3m/4),Γ=-1.0)


# apply it to the flow
apply!(f, sim.flow.u); plot()
sim.flow.u[:,:,1] .+= 0.5
sim.flow.u[:,:,2] .+= 0.1
vector_plot!(sim.flow.u)
flood(sim.flow.u[:,:,1]; shift=(-0.5,-0.5))
flood(sim.flow.u[:,:,2]; shift=(-0.5,-0.5))

# compute vorticity
sim.flow.σ .= 0.0
@inside sim.flow.σ[I] =-WaterLily.curl(3,I,sim.flow.u)
flood(sim.flow.σ; shift=(-0.5,-0.5))

# compute streamfunction from vorticity
Helmoltz!(uϕ,uψ,ϕ,ψ,sim.flow)

# error
R = inside(ψ)

flood(ϕ; shift=(-0.5,-0.5)); vector_plot!(uϕ)
flood(ψ; shift=(-0.5,-0.5)); vector_plot!(uψ)

# println("L₂-norm error u-velocity ", WaterLily.L₂(u[R,1].-sim.flow.u[R,1]))
# println("L₂-norm error v-velocity ", WaterLily.L₂(u[R,2].-sim.flow.u[R,2]))
@inside sim.flow.σ[I] = WaterLily.div(I,uψ); flood(sim.flow.σ)
@inside sim.flow.σ[I] = WaterLily.curl(3,I,uϕ); flood(sim.flow.σ)

# check
flood(uψ[R,1].+uϕ[R,1].-sim.flow.u[R,1]; shift=(-0.5,-0.5))
flood(uψ[R,2].+uϕ[R,2].-sim.flow.u[R,2]; shift=(-0.5,-0.5))

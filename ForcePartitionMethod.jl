using WaterLily
using Plots; gr()
using StaticArrays
include("examples/TwoD_plots.jl")
function vector_plot!(u::AbstractArray)
    x,y=repeat(axes(u,1),outer=length(axes(u,2))),repeat(axes(u,2),inner=length(axes(u,1)))
    us = u.*.√sum(u.^2,dims=length(size(u)));
    quiver!(x,y,quiver=([us[:,:,1]...],[us[:,:,2]...]))
end
norm(I,u) = √sum(u[I,:].^2)
dot(I::CartesianIndex{n},a,b) where n = sum(ntuple(i->a[I,i]*b[I,i],n))
cross(I::CartesianIndex{3},a,b) = WaterLily.fSV(i->WaterLily.permute((j,k)->a[I,j]*b[I,k],i),3)
cross(I::CartesianIndex{2},a,b) = WaterLily.fSV(i->-a[I]*b[I,i%2+1],2)
function fmpm(flow,pois,i)
    
end

# parameters
Re = 250
U = 1
nx = ny = 2^8
radius, center = ny/8, ny/2
L = nx
center = SA[center,center]

# make a body
circle = AutoBody((x,t)->√sum(abs2, x .- center .- 1.5) - radius)

# generate sim
sim = Simulation((nx,ny), (U,0), radius; ν=U*radius/Re, body=circle)

# the boundary value is the surface normal
function normal_x(x,i)
    dᵢ,nᵢ,_ = measure(sim.body,x,0.0)
    μ₀ = WaterLily.μ₀(dᵢ,1)
    (1.0-μ₀)*nᵢ[i] # x-component of the normal
end
ϕ = copy(sim.flow.p)
∇ϕ = copy(sim.flow.u)

# generate source term
apply!(x->normal_x(x,1),ϕ); BC!(ϕ)
flood(ϕ,levels=21,clims=(-1,1))

# solve for the potential 
μ₀ = copy(sim.flow.μ₀); #μ₀ .= 1; BC!(μ₀,[0,0])
pois = WaterLily.Poisson(sim.flow.p,μ₀,sim.flow.σ)
@inside pois.z[I] = ϕ[I]
solver!(pois); ϕ.=pois.x; BC!(ϕ)

# compute gradient of potential -> velocity
for i ∈ 1:2
    @WaterLily.loop ∇ϕ[I,i] -= μ₀[I,i]*WaterLily.∂(i,I,ϕ) over I ∈ inside(pois.x)
end

# plot to check
R = inside(ϕ)
flood(ϕ[R],shift=(-1.,-1.),clims=(-radius,radius),levels=11,filled=true)
flood(∇ϕ[R,1],levels=11,filled=true)
body_plot!(sim)

# we can check by simply multyplying the potential by the laplacian and
# we should get back the source term
# @inside pois.x[I] = WaterLily.mult(I,pois.L,pois.D,ϕ)
# flood(pois.x,shift=(-1.,-1.),levels=51,filled=true)
# body_plot!(sim)

# compute vorticity once
ω = copy(sim.flow.σ); @inside ω[I] = WaterLily.curl(3,I,sim.flow.u)

function normal(i,x)
    _,nᵢ,_ = measure(sim.body,x,0.0); nᵢ[i] # i-component of the normal
end
n = copy(sim.flow.u); apply!(normal,n); BC!(n)
nⱼ = @view n[:,:,2]

function ndotdUdt(I,t,body,n)
    x = WaterLily.loc(0,I)
    # The velocity depends on the material change of ξ=m(x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x->body.map(x,t), x)
    dot = ForwardDiff.derivative(t->body.map(x,t), t)
    v = -J\dot; a=0.0#?
    return sum(n[I,:].*a)
end

# body forces
@inside sim.flow.σ[I] = ndotdUdt(I,0.0,sim.body,n)*ϕ[I] - 0.5*norm(I,sim.flow.V)^2*nⱼ[I]
flood(sim.flow.σ, clims=(-1,1))
Ck = WaterLily.∮nds(sim.flow.σ,sim.flow.f,sim.body,0)

# vorticity forces
# uᵥ,uϕ = helmholtz(sim.flow.u);
@inside sim.flow.σ[I] = (0.5*dot(I,uᵥ,uᵥ)+dot(I,uᵩ,uᵥ))*ϕ[I]
for i ∈ 1:n
    @WaterLily.loop sim.flow.f[I,i] = WaterLily.∂(i,I,sim.flow.σ) over I in inside(ϕ)
end
@WaterLily.loop sim.flow.f[I,:] .+= cross(I,ω,sim.flow.u) over I in inside(ϕ)
@inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.f)*ϕ[I]
flood(sim.flow.σ, clims=(-1,1))
Cω = sum(sim.flow.σ[inside(sim.flow.σ)])

# viscous force
@WaterLily.loop sim.flow.f[I,:] = cross(I,ω,n) over I in inside(ϕ)
∇ϕ[:,:,1] .-= 1.0
@inside sim.flow.σ[I] = dot(I,sim.flow.f,∇ϕ)
∇ϕ[:,:,1] .+= 1.0
flood(sim.flow.σ, clims=(-1,1))
Cσ = sim.flow.ν.*WaterLily.∮nds(sim.flow.σ,sim.flow.f,sim.body,0)

# potential force
@inside sim.flow.σ[I] = 0.5*dot(I,uᵩ,uᵩ)*ϕ[I]
for i ∈ 1:n
    @WaterLily.loop sim.flow.f[I,i] = WaterLily.∂(i,I,sim.flow.σ) over I in inside(ϕ)
end
@inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.f)
Cϕ = sum(sim.flow.σ[inside(sim.flow.σ)])

# boundary force
N,n = WaterLily.size_u(sim.flow.u)
dudt = copy(sim.flow.u); C∑=0.0# must be filled manually
# for i ∈ 1:n
#     n=zero(2); n[i]=1
#     for I ∈ slice(N,1,i)
#         C∑ += sim.flow.ν*WaterLilycross(I,ω,n)*∇ϕ[I,i]
#         C∑ += sim.flow.ν*cross(I+N[i]*δ(i,I),ω,-n)*∇ϕ[I+N[i]*δ(i,I),i]
#         C∑ -= dudt[I,i]*n[I,i]*ϕ[I]
#     end
end

# helmholtz decomposition http://pcmap.unizar.es/~jaca2016/PDFXII/Ahusborde.pdf
# function uΨλ(i,xy)
#     x,y = @. (xy-1.5)*2π/L  
#     i==1 && return  sin(x)*cos(y)
#     i==2 && return -sin(y)*cos(x)
# end
# function uΦλ(i,xy)
#     x,y = @. (xy-1.5)*2π/L  
#     i==1 && return sin(y)*cos(x)
#     i==2 && return sin(x)*cos(y)
# end
# function uHλ(i,xy)  
#     i==1 && return 0.5
#     -1.0
# end

# uΨ = copy(sim.flow.u); apply!(uΨλ,uΨ)
# uΦ = copy(sim.flow.u); apply!(uΦλ,uΦ)
# uH = copy(sim.flow.u); apply!(uHλ,uH)

# u = copy(sim.flow.u); u .= uΨ .+ uΦ .+ uH
# @inside sim.flow.σ[I] = WaterLily.curl(3,I,u)*sim.L/sim.U
# flood(sim.flow.σ)
# vector_plot!(u)
# 

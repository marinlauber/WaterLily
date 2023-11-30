using WaterLily
using Plots; gr()
using StaticArrays
include("examples/TwoD_plots.jl")
function vector_plot!(u::AbstractArray)
    x,y=repeat(axes(u,1),outer=length(axes(u,2))),repeat(axes(u,2),inner=length(axes(u,1)))
    us = u.*.√sum(u.^2,dims=length(size(u)));
    quiver!(x,y,quiver=([us[:,:,1]...],[us[:,:,2]...]))
end
function fmpm(flow,pois,i)
    #
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
    @WaterLily.loop ∇ϕ[I,i] -= μ₀[I,i]*WaterLily.∂(i,I,ϕ) over I ∈ WaterLily.inside(pois.x)
end

# plot to check
R = inside(ϕ)
flood(ϕ[R],shift=(-1.,-1.),clims=(-radius,radius),levels=51,filled=true)
flood(∇ϕ[R,1],levels=51,filled=true)
body_plot!(sim)

# we can check by simply multyplying the potential by the laplacian and
# we should get back the source term
# @inside pois.x[I] = WaterLily.mult(I,pois.L,pois.D,ϕ)
# flood(pois.x,shift=(-1.,-1.),levels=51,filled=true)
# body_plot!(sim)

function normal(x,i)
    _,nᵢ,_ = measure(sim.body,x,0.0); nᵢ[i] # i-component of the normal
end
nⱼ = copy(ϕ); component=2; apply!(x->normal(x,2),nⱼ); BC!(nⱼ)

# viscous force
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
@inside sim.flow.σ[I] = sim.flow.ν.*(sim.flow.σ[I]*nⱼ[I]).*(∇ϕ[I,1] .- 1)
flood(sim.flow.σ, clims=(-1,1))
Cσ = WaterLily.∮nds(sim.flow.σ,sim.flow.f,sim.body,0)

dot(I::CartesianIndex{n},a,b) where n = sum(ntuple(i->a[I,i]*b[I,i],n))
cross(I::CartesianIndex{3},a,b) = WaterLily.fSV(i->WaterLily.permute((j,k)->a[I,j]*b[I,k],i),3)
cross(I::CartesianIndex{2},a,b) = WaterLily.fSV(i->-a[I]*b[I,i%2+1],2)

# vorticity forces
ωxu = copy(sim.flow.u);
# uᵥ,uϕ = helmholtz(sim.flow.u);
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
@WaterLily.loop ωxu[I,:] .= cross(I,sim.flow.σ,sim.flow.u) over I in inside(ϕ)
@inside sim.flow.σ[I] = WaterLily.div(I,ωxu)*ϕ[I]
flood(sim.flow.σ, clims=(-1,1))
Cω = sum(sim.flow.σ[inside(sim.flow.p)])

# boundary force
@inside sim.flow.σ[I] = sim.flow.ν*WaterLily.curl(3,I,sim.flow.u)
N,n = Waterlily.size_u(sim.flow.u)
for i ∈ 1:n
    n=zero(2); n[i]=1
    for I ∈ slice(N,1,j)
        sum += cross(I,sim.flow.σ,n) + corss(I+δ(j,I),sim.flow.σ,-n)
    end
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

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
nx = ny = 2^5
radius, center = ny/4, ny/2
L = nx
center = SA[center,center]

# make a body
circle = AutoBody((x,t)->√sum(abs2, x .- center .- 1.5) - radius)

# generate sim
sim = Simulation((nx,ny), (U,0), radius; ν=U*radius/Re, body=circle)

component = 1

# the boundary value is the surface normal
function normal_x(x)
    dᵢ,nᵢ,_ = measure(sim.body,x,0.0)
    μ₀ = WaterLily.μ₀(dᵢ,1)
    (1.0-μ₀)*nᵢ[component] # x-component of the normal
end
ϕ = copy(sim.flow.p)
∇ϕ = copy(sim.flow.u)

# generate source term
apply!(normal_x,ϕ); BC!(ϕ)
flood(ϕ)

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
flood(clamp.(ϕ[:,:],-15,15),shift=(-1.,-1.),levels=51,filled=true)
flood(∇ϕ[inside(sim.flow.p),component],levels=51,filled=true)
body_plot!(sim)

# we can check by simply multyplying the potential by the laplacian and
# we should get back the source term
# @inside pois.x[I] = WaterLily.mult(I,pois.L,pois.D,ϕ)
# flood(pois.x,shift=(-1.,-1.),levels=51,filled=true)
# body_plot!(sim)

# helmholtz decomposition http://pcmap.unizar.es/~jaca2016/PDFXII/Ahusborde.pdf
function uΨλ(i,xy)
    x,y = @. (xy-1.5)*2π/L  
    i==1 && return  sin(x)*cos(y)
    i==2 && return -sin(y)*cos(x)
end
function uΦλ(i,xy)
    x,y = @. (xy-1.5)*2π/L  
    i==1 && return sin(y)*cos(x)
    i==2 && return sin(x)*cos(y)
end
function uHλ(i,xy)  
    i==1 && return 0.5
    -1.0
end

uΨ = copy(sim.flow.u); apply!(uΨλ,uΨ)
uΦ = copy(sim.flow.u); apply!(uΦλ,uΦ)
uH = copy(sim.flow.u); apply!(uHλ,uH)

u = copy(sim.flow.u); u .= uΨ .+ uΦ .+ uH
@inside sim.flow.σ[I] = WaterLily.curl(3,I,u)*sim.L/sim.U
flood(sim.flow.σ)
vector_plot!(u)
# 

using WaterLily
using Plots; gr()
using StaticArrays
include("examples/TwoD_plots.jl")

function fmpm(flow,pois,i)
    #
end

# parameters
Re = 250
U = 1
nx = ny = 2^5
radius, center = ny/4, ny/2

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

# generate source term
apply!(normal_x,ϕ)
BC!(ϕ)
flood(ϕ)

# solve for the potential 
μ₀ = copy(sim.flow.μ₀); #μ₀ .= 1; BC!(μ₀,[0,0])
pois = WaterLily.Poisson(sim.flow.p,μ₀,sim.flow.σ)
@inside pois.z[I] = ϕ[I]
solver!(pois)
ϕ[:,:] = pois.x[:,:]#./radius^2
BC!(ϕ)
for i ∈ 1:2  # apply pressure solution b.x
    @WaterLily.loop sim.flow.u[I,i] -= pois.L[I,i]*WaterLily.∂(i,I,ϕ) over I ∈ WaterLily.inside(pois.x)
end
flood(clamp.(ϕ[:,:]./radius^2,-0.25,0.25),levels=51,filled=true)
flood(sim.flow.u[inside(sim.flow.p),component],levels=51,filled=true)
body_plot!(sim)

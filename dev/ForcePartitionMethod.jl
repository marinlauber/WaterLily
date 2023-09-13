using WaterLily
using Plots; gr()
using StaticArrays
include("../examples/TwoD_plots.jl")

norm(v) = √(v'*v)

# parameters
Re = 250
U = 1
nx = ny = 2^5
radius, center = ny/4, ny/2
duration = 5
step = 0.1

center = SA[center,center]

# make a body
circle = AutoBody((x,t)->√sum(abs2, x .- center .- 1.5) - radius)

# generate sim
sim = Simulation((nx,ny), (U,0), radius; ν=U*radius/Re, body=circle)

# function my_function(i,x)
#     θ = atan((x[2]-center[2])/(x[1]-center[1]))
#     r = √((x[1]-center[1])^2+(x[2]-center[2])^2)
#     return r*(1+(radius/r)^2)*cos(θ)
# end
function normal_x(x)
    m = x[1]-center[1]-1.5
    return m/max(1e-9, norm(x-center.-1.5))
end
# # get start time
# t₀ = round(sim_time(sim))
# @time @gif for tᵢ in range(t₀,t₀+duration;step)
#     # update until time tᵢ in the background, equivalent to
#     sim_step!(sim,tᵢ;remeasure=false)
# end
# flood(sim.flow.p; shift=(-0.5,-0.5),clims=(-1,1))

# potential
# ϕ = copy(sim.flow.p)
# apply!(normal_x,sim.flow.u)
# flood(sim.flow.u[:,:,1])
# sim.flow.p .= 0.0
# sim.flow.σ .= 0.0
# pois = WaterLily.Poisson(sim.flow.p,sim.flow.μ₀,sim.flow.σ)
# # set source term
# @inside pois.z[I] = sim.flow.u[I,1]
# solver!(pois)
# flood(pois.z[:,:])
# kill
# analytical potential
ϕ = copy(sim.flow.p)
# ψ = copy(sim.flow.u)
# apply!(my_function,sim.flow.u)
# sim.flow.u[:,:,1] = sim.flow.u[:,:,1].*sim.flow.μ₀[:,:,1]
# flood(sim.flow.u[:,:,1])
# @inside ϕ[I] = WaterLily.∂(1,I,sim.flow.u)
# ψ[:,:,1] = ϕ[:,:]
# @inside ϕ[I] = WaterLily.∂(2,I,sim.flow.u)
# ψ[:,:,2] = ϕ[:,:]
# @inside ϕ[I] = WaterLily.div(I,ψ)
# flood(ϕ[:,:],clim=(-24,24))


# this is combersome
# apply!(normal_x,ψ)
apply!(normal_x,ϕ)
μ₀ = copy(sim.flow.μ₀)
BC!(μ₀,[1,1])
# ϕ.*=μ₀[:,:,1]
BC!(ϕ)
flood(ϕ)

# uniform 

# sim.flow.σ .= 0.0
pois = WaterLily.Poisson(sim.flow.p,sim.flow.μ₀,sim.flow.σ)
# set source term

@inside ϕ[I] = ϕ[I]
BC!(ϕ)
@inside pois.z[I] = ϕ[I]

# WaterLily.residual!(pois)
# display(WaterLily.L₂(pois))

solver!(pois)
ϕ[:,:] = pois.x[:,:]./radius^2
BC!(ϕ)
# fₓ = WaterLily.∮nds(ϕ,sim.flow.f,circle)
# println(fₓ/radius)
flood(ϕ[:,:],levels=51,filled=true)
# for i in 1:2
#     @WaterLily.loop sim.flow.u[I,i] -= μ₀[I,i]*WaterLily.∂(i,I,ϕ) over I ∈ WaterLily.inside(ϕ)
# end

# flood(sim.flow.u[inside(sim.flow.p),1])


# function ϕₑ(x)
#     r = max(1e-4, norm(x.-center.-1.5))
#     return r*(1+radius^2/r^2)*cos(atan((x[2]-center[2]),(x[1]-center[1])))
# end
# apply!(ϕₑ,ϕ)
# flood(ϕ.*μ₀[:,:,1], levels=21)

# # potential flow from initial step
# apply!((i,x)-> i==1 ? 1.0 : 0.0,sim.flow.u)
# sim.flow.u .*= sim.flow.μ₀
# BC!(sim.flow.u,[1.0,0.0])
# @inside pois.z[I] = WaterLily.div(I,sim.flow.u)
# solver!(pois)
# for i in 1:2
#     @WaterLily.loop sim.flow.u[I,i] -= pois.L[I,i]*WaterLily.∂(i,I,pois.x) over I ∈ WaterLily.inside(pois.x)
# end
# flood(sim.flow.u[:,:,2])
# contour!(sim.flow.u[:,:,2], levels=21)
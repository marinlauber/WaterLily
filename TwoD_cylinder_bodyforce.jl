using WaterLily
using LinearAlgebra: norm
function circle(L=64;Re=200,U=1,g=(0.,0.))
    radius, center = L/2, L
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((2L,2L), (U,0), radius; ν=U*radius/Re, body, exitBC=true, g)
end

include("examples/TwoD_plots.jl")
sim = circle(g=(0.0,0.))
a = sim.flow; b=sim.pois

a.u⁰ .= a.u; WaterLily.scale_u!(a,0)
# predictor u → u'
WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν,g=a.g)
WaterLily.BDIM!(a); BC!(a.u,a.U,a.exitBC)
a.exitBC && WaterLily.exitBC!(a.u,a.u⁰,a.U,a.Δt[end]) # convective exit
WaterLily.project!(a,b); BC!(a.u,a.U,a.exitBC)

# # intialize
# t₀ = sim_time(sim); duration = 10; tstep = 0.1

# # step and plot
# @time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
#     # update until time tᵢ in the background
#     sim_step!(sim,tᵢ)
  
#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
#     a = sim.flow.σ; @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#     flood(a[inside(a)],clims=(-10,10)); body_plot!(sim)
# end
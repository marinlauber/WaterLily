using WaterLily
using LinearAlgebra: norm
using StaticArrays
function circle(n,m;Re=1e4,U=1)
    radius, center = m/8, m/2; Sr=5
    function map(xy,t)
        Ω = Sr*U/(2π*radius)
        SA[cos(Ω*t) sin(Ω*t); -sin(Ω*t) cos(Ω*t)] * (xy.-center)
    end
    body = AutoBody((x,t)->√sum(abs2, x) - radius, map)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body)
end

include("TwoD_plots.jl")
sim = circle(3*2^6,2^7)

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1

# step and plot
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ; @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(a[inside(a)],clims=(-10,10)); body_plot!(sim)
end
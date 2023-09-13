using WaterLily
using StaticArrays
using ParametricBodies # New package
using Plots
include("TwoD_plots.jl")
function make_sim(nb=3;L=32,Re=1e3,λ=2.0,Π₃=0.25,U=1,n=8,m=6,T=Float32,mem=Array)
    R = Int(L/2Π₃); ω = T(λ*U/R) # ensure type stable
    origin,pivot = SA[0.25f0n*R,0.5f0m*R],SA[0.125f0R,1f0R]
    # α(t) = 0.0f0
    function map(x,t)
        ξ = x-origin; θ,r = atan(ξ[2],ξ[1]),√sum(abs2, ξ) # cylindrical CS
        θ = mod(θ+3f0π/4f0+ω*t,2f0π/nb)-3f0π/4f0 # make cylindrical-periodic
        x = SA[r*cos(θ),r*sin(θ)] # map back to cartesian CS
        # R = SA[cos(α(t)) sin(α(t)); -sin(α(t)) cos(α(t))] # pitch rotation matrix
        ξ = x+pivot # move to origin and align with x-axis
        return SA[ξ[1],abs(ξ[2])]   # reflect to positive y
    end

    # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = 0.5f0L*SA[(1-s)^2,NACA(1-s)]
    body = ParametricBody(foil,(0,1);map,T,mem)

    Simulation((n*R,m*R),(U,0),L;ν=ω*R*L/Re,body,T,mem)
end

# make a simulation
sim = make_sim(;L=32);

# intialize
t₀ = sim_time(sim); duration = 20.0; tstep = 0.1

# step and plot
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ; @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(a[inside(a)],clims=(-20,20)); body_plot!(sim)
end
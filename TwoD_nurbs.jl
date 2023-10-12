using WaterLily
using StaticArrays
include("../examples/TwoD_plots.jl")

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(clamp.(ω,-limit,limit)',dpi=300,
    color=palette(:RdBu_11), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)

# parameters
L=2^5
Re=250
U=1;amp=π/4
ϵ=0.5
thk=2ϵ+√2

# make sim
Points = 3L.+0.5*L*Array(reduce(hcat,[SVector(i-5, sin(i^2)) for i in 1:9])')
# Points = 3L.+0.5*L*Array(reduce(hcat,[SVector(i-5, sin(i^2)) for i in 1:9])')
# Points = Array([[3L,3L] [2L,3L] [3L,4L]]')
Body = SplineBody(Points;deg=2,thk=thk,compose=false,T=Float64)
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)

# sim
t₀ = round(sim_time(sim))
duration = 15; tstep = 0.1;
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
    Plots.plot!(Body.pts[:,1],Body.pts[:,2],markers=:o,legend=false)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
using WaterLily
include("examples/TwoD_plots.jl")

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
Body = AutoBody((x,t)->√sum(abs2, x .- [3L,3L]) - L/2)
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)

t₀ = round(sim_time(sim))
duration = 15; tstep = 0.1; force = []
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)

    # compute forces 
    fi = WaterLily.∮nds(sim.flow.p,sim.flow.f,Body)
    push!(force,fi)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
forces = reduce(hcat,force)
gif(anim, "circle_flow.gif", fps=24)

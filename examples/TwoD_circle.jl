using WaterLily
function circle(n,m;Re=250,U=1)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body)
end

include("TwoD_plots.jl")
sim = circle(3*2^6,2^7)

function Helmholtz!(ϕ,ω,a::Flow{n},b::AbstractPoisson) where n
    dt = sim.flow.Δt[end]
    for i ∈ 1:n 
        @WaterLily.loop ϕ[I,i] -= dt*b.L[I,i]*WaterLily.∂(i,I,b.x) over I ∈ inside(b.x)
    end
    # vorticity component is diff between potential component and complete component
    ω .= a.u .- ϕ
end

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1
uϕ = copy(sim.flow.u); uω = copy(sim.flow.u);

# step and plot
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)
    Helmholtz!(uϕ,uω,sim.flow,sim.pois)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ;
    # @inside a[I] = WaterLily.div(I,sim.flow.u)
    @inside a[I] = WaterLily.curl(3,I,uϕ)*sim.L/sim.U
    flood(a[inside(a)],clims=(-5,5)); body_plot!(sim)
    # flood(sim.flow.p[inside(a)],clims=(-1,1)); body_plot!(sim)
    # flood(uω[inside(a),1],clims=(-5,5)); body_plot!(sim)
end
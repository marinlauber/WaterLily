using WaterLily,StaticArrays
function hover(L=2^5;Re=250,U=1,ϵ=1.0,thk=2ϵ+√2,T=Float32)
    # Line segment SDF
    function sdf(x,t)
        √sum(abs2,x .- SA[0,clamp(x[2],-L/2,L/2)])-thk/2
    end
    # Oscillating motion and rotation
    function map(x,t)
        x - SA[6L-L*sin(t*U/L),6L]
    end
    Simulation((12L,12L),(0,0),L;U,ν=U*L/Re,body=AutoBody(sdf,map),ϵ,T)
end
function cicrle(L=2^5;Re=250,U=1,ϵ=1.0,thk=2ϵ+√2,T=Float32)
    # circle SDF
    function sdf(x,t)
        √sum(abs2, x .- 3L) - L/2
    end
    Simulation((10L,6L),(1,0),L;U,ν=U*L/Re,body=AutoBody(sdf),ϵ,T)
end
# sim = hover(;T=Float64);
sim = cicrle(;T=Float64);
include("TwoD_plots.jl")
duration=20;
R=inside(sim.flow.p)
drag=[]; probe=[];
anim = @animate while sim_time(sim)<duration
    measure!(sim); mom_step!(sim.flow,sim.pois)
    @show length(sim.pois.levels)
    forces = 2WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,WaterLily.time(sim))/sim.L
    push!(drag,forces[1]); push!(probe,sim.flow.p[32,32])
    @inside sim.flow.σ[I] = sim.flow.p[I]
    # flood(sim.flow.σ[R],clims=(-1,1)); body_plot!(sim)
    a = sim.flow.σ;
    p1=Plots.contourf(axes(a,1),axes(a,2),clamp.(a',-1,1),linewidth=0,levels=10,
                   color=:RdBu_11,clims=(-1,1),aspect_ratio=:equal)
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!(p1,sim.flow.σ[R]';levels=[0],lines=:black)
    p2 = plot(reshape(copy(sim.pois.n),(2,:))',ylim=(0,32))
    p3 = plot(drag,ylim=(-2,0))
    plot(p1,p2,p3, layout=@layout [a ; b c])
    println("tU/L=",round(sim_time(sim),digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
gif(anim, "result_$(eltype(sim.flow.u)).gif", fps = 15)
# plot(reshape(sim.pois.n',(:,2)),ylim=(0,32))
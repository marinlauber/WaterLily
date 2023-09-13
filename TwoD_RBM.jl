using WaterLily
using StaticArrays
using Plots
include("examples/TwoD_plots.jl")

CI = WaterLily.CI
inside_u = WaterLily.inside_u
ϕ = WaterLily.ϕ
∂ = WaterLily.∂
ϕu = WaterLily.ϕu
@fastmath function mom_step_g!(a::Flow,b::AbstractPoisson;g=[0.,0.,0.])
    a.u⁰ .= a.u; a.u .= 0;
    N,n = WaterLily.size_u(a.u)
    dt = a.Δt[end]
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    WaterLily.BDIM!(a);
    for i ∈ 1:n
        @WaterLily.loop a.u[I,i] = a.u[I,i] + g[i] over I ∈ inside_u(N,i)
    end
    BC!(a.u,a.U)
    WaterLily.project!(a,b); BC!(a.u,a.U)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a);
    for i ∈ 1:n
        @WaterLily.loop a.u[I,i] = a.u[I,i] + g[i] over I ∈ inside_u(N,i)
    end
    BC!(a.u,a.U,2)
    WaterLily.project!(a,b,2); a.u ./= 2; BC!(a.u,a.U)
    push!(a.Δt,WaterLily.CFL(a))
end
function WaterLily.conv_diff!(r,u,Φ;ν=0.1,g=[0.,0.,0.])
    r .= 0.
    N,n = WaterLily.size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        @WaterLily.loop r[I,i] += ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u)-ν*∂(j,CI(I,i),u) over I ∈ WaterLily.slice(N,2,j,2)
        @WaterLily.loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u))-ν*∂(j,CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @WaterLily.loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        @WaterLily.loop r[I-δ(j,I),i] += - ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u) + ν*∂(j,CI(I,i),u) over I ∈ WaterLily.slice(N,N[j],j,2)
        # i==j && @WaterLily.loop r[I,i] += g[i] over I ∈ inside_u(N,i)
    end
    for i ∈ 1:n
        @WaterLily.loop r[I,i] += g[i] over I ∈ inside_u(N,i)
    end
end
function circle(c=0,p=4;St=0.3,Re=250,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    L=2^p
    center,r = SA[3L,3L], L
    ν = U*L/Re
    f = St*U/L

    # make a body
    norm2(x) = √sum(abs2,x)
    body = AutoBody() do xyz,t
        x,y = xyz - center
        norm2(SA[x,y+c*L*sin(2π*f*t)])-r
    end

    Simulation((8L,6L),(0,0),L;ν,U,body,mem)
end

sim = circle();
# sim = circle(1.0);

# intialize
t₀ = sim_time(sim)
duration = 10
tstep = 0.2
St=0.3; f = St*sim.U/sim.L

# step and write
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    # sim_step!(sim,tᵢ,remeasure=false)
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ*sim.L/sim.U
        # update the body
        # mom_step_g!(sim.flow,sim.pois;g=[0.,sim.flow.Δt[end]*-2π*f*sim.L*cos(2π*f*t)])
        mom_step_g!(sim.flow,sim.pois;g=[t<2sim.L ? 0.5 : 0.0,0.0])
        # measure!(sim,t); mom_step!(sim.flow,sim.pois)
        # finish the time step
        t += sim.flow.Δt[end]
    end

    # print time step
    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end


using WaterLily
using LinearAlgebra: norm
function circle(n,m;Re=250,U=1)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body)
end
# function Helmholtz!(uϕ,uψ,a::Flow{n},b::AbstractPoisson) where n
#     # source term of the Poisson problem
#     # @inside b.z[I] = -WaterLily.curl(3,I,a.u)
#     @inside b.z[I] = WaterLily.div(I,a.u)
#     # solve for the stream function
#     solver!(b); BC!(b.x)
#     # compute curl of stream function
#     # @WaterLily.loop uψ[I,1] = WaterLily.∂(2,I,b.x) over I ∈ inside(b.x)
#     # @WaterLily.loop uψ[I,2] =-WaterLily.∂(1,I,b.x) over I ∈ inside(b.x)
#     # compute grad(potential) part
#     # uψ.*=b.L;
#     # for i ∈ 1:n
#     #     uϕ[:,:,i] .= a.u[:,:,i] .- uψ[:,:,i] .- a.U[i]
#     # end
#     for i ∈ 1:n  # apply pressure solution b.x
#         @WaterLily.loop uϕ[I,i] = b.L[I,i]*WaterLily.∂(i,I,b.x) over I ∈ inside(b.x)
#         @WaterLily.loop uψ[I,i] = a.u[I,i] - uϕ[I,i] over I ∈ inside(b.x)
#     end
#     BC!(uϕ,[0.,0.]); BC!(uψ,[0.,0.])
# end
# function Helmholtz!(uϕ,uψ,a::Flow{n})
#     c=copy(a.p); d=copy(a.p);
#     # source term of the Poisson problem
#     @inside c[I] = -WaterLily.curl(3,I,a.u)
#     @inside d[I] =  WaterLily.div(I,a.u)
#     # solve for the stream function and potential
# end
function Integrate!(ψ,ϕ,ω,div)
    ψ .= 0.0; ϕ .= 0.0
    for I ∈ CartesianIndices(ψ)
        @WaterLily.loop (ψ[I] += G∞(I,J)*ω[J]; ϕ[I] += G∞(I,J)*div[J]) over J ∈ CartesianIndices(ω)
    end
end
function Integrate!(ϕ,div)
    ϕ .= 0.0
    for I ∈ CartesianIndices(ψ)
        @WaterLily.loop ϕ[I] += G∞(I,J)*div[J] over J ∈ CartesianIndices(ω)
    end
end
function G∞(I,J) # ψ and ω is on the edge of the cell
    I==J && return 0.0 # ignore self contribution
    return log(norm(loc(0,I).-loc(0,J)))/2π
end
function Helmholtz!(ϕ,ψ,a)
    # compute vorticity
    # @inside a.pˢ[I] =-WaterLily.curl(3,I,a.u)
    # compute divergence
    @inside a.σ[I] = WaterLily.div(I,a.u)
    # compute streamfunction and potential at once
    # Integrate!(ψ,ϕ,a.pˢ,a.σ);
    Integrate!(ϕ,a.σ);
end
function Helmholtz!(uϕ,uψ,ϕ,ψ,a::Flow{n}) where n
    Helmholtz!(ϕ,ψ,a)
    for i ∈ 1:n
        j = i%2+1 # other component
        # compute gradient of potential
        @WaterLily.loop uϕ[I,i] = WaterLily.∂(i,I,ϕ) over I ∈ inside(ϕ)
        # compute velocity from stream function, ψ is edge-centered
        # @WaterLily.loop uψ[I,i] = sign(j-i)*WaterLily.∂(j,I+δ(j,I),ψ) over I ∈ inside(ψ)
    end
    uψ .= sim.flow.u .- uϕ
end

include("TwoD_plots.jl")
sim = circle(3*2^5,2^6)

# intialize
t₀ = sim_time(sim); duration = 1; tstep = 0.1
uϕ = zero(sim.flow.u); uω = zero(sim.flow.u);
ϕ = zero(sim.flow.p); ω = zero(sim.flow.p);
# pois = MultiLevelPoisson(copy(sim.flow.p),sim.flow.μ₀,sim.flow.σ)

# # single momentum step
# mom_step!(sim.flow,sim.pois)
# Helmholtz!(uϕ,uω,ϕ,ω,sim.flow)

# # plot
# flood(uϕ[inside(sim.flow.σ),1],clims=(-1,1)); body_plot!(sim)
# flood(uϕ[inside(sim.flow.σ),2],clims=(-1,1)); body_plot!(sim)
# flood(uω[inside(sim.flow.σ),1],clims=(-1,1)); body_plot!(sim)
# flood(uω[inside(sim.flow.σ),2],clims=(-1,1)); body_plot!(sim)


# step and plot
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)
    Helmholtz!(uϕ,uω,ϕ,ω,sim.flow)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ;
    # @inside a[I] = WaterLily.div(I,uω)
    # @inside a[I] = WaterLily.curl(3,I,uϕ)
    # @inside a[I] = WaterLily.curl(3,I,uϕ.+uω)
    @inside a[I] = WaterLily.div(I,uϕ.+uω)
    flood(a[inside(a)],clims=(-1,1)); body_plot!(sim)
end
using StaticArrays
using ForwardDiff
using LinearAlgebra: ×, tr, norm # can this be an issue?
using WaterLily: kern, ∂, inside_u, AbstractBody
using WaterLily
include("TwoD_plots.jl")

# viscous stress tensor, 
∇²u(I::CartesianIndex{2},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(I::CartesianIndex{3},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:3, j ∈ 1:3]
"""normal componenent integration kernel"""
@inline function nds(body::AbstractBody,x,t)
    d,n,_ = measure(body,x,t)
    n*WaterLily.kern(clamp(d,-1,1))
end
"""moment kernel"""
@inline function xnds(body::AbstractBody,x₀::SVector{N,T},x,t,ϵ) where {N,T}
    (x-x₀)×nds_ϵ(body,x,t,ϵ)
end
"""surface integral of pressure"""
function ∮nds_ϵ(p::AbstractArray{T,N},df::AbstractArray{T},body::AutoBody,t=0,ε=1.) where {T,N}
    @WaterLily.loop df[I,:] = p[I]*nds_ϵ(body,loc(0,I,T),t,ε) over I ∈ inside(p)
    [sum(@inbounds(df[inside(p),i])) for i ∈ 1:N] |> Array
end
"""curvature corrected kernel evaluated ε away from the body"""
@inline function nds_ϵ(body::AbstractBody,x,t,ε)
    d,n,_ = measure(body,x,t); κ = 0.5tr(ForwardDiff.hessian(y -> body.sdf(y,t), x))
    κ = isnan(κ) ? 0. : κ;
    n*WaterLily.kern(clamp(d-ε,-1,1))/prod(1.0.+2κ*d)
end
"""for lack of a better name, this is the surface integral of the velocity"""
function diagnostics(a::Simulation,x₀::SVector{N,T}) where {N,T}
    # get time
    t = WaterLily.time(a); Nu,n = WaterLily.size_u(a.flow.u); Inside = CartesianIndices(map(i->(2:i-1),Nu)) 
    # compute pressure  and viscous contributions
    @WaterLily.loop a.flow.f[I,:] .= -a.flow.p[I]*nds_ϵ(a.body,loc(0,I,T),t,a.ϵ) over I ∈ inside(a.flow.p)
    @WaterLily.loop a.flow.f[I,:] .+= a.flow.ν*∇²u(I,a.flow.u)*nds_ϵ(a.body,loc(0,I,T),t,a.ϵ) over I ∈ inside(Inside)
    # integrate the pressure force
    force=[sum(@inbounds(a.flow.f[inside(a.flow.p),i])) for i ∈ 1:N] |> Array
    # compute pressure moment contribution
    @WaterLily.loop a.flow.σ[I] = -a.flow.p[I]*xnds(a.body,x₀,loc(0,I,T),t,a.ϵ) over I ∈ inside(a.flow.p)
    # integrate moments
    moment=sum(@inbounds(a.flow.σ[inside(a.flow.p)]))
    return force,moment
end

using StaticArrays
# viscous stress tensor,using StaticArrays avoid allocation of memory is efficient for tensor-vector operations
∇²u(I::CartesianIndex{2},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(I::CartesianIndex{3},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:3, j ∈ 1:3]
"""
    ∮τnds(u::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0)

Compute the viscous force on a immersed body. 
"""
function ∮τnds(u::AbstractArray{T,N},df::AbstractArray{T,N},body::AbstractBody,t=0) where {T,N}
    Nu,_ = WaterLily.size_u(u); In = CartesianIndices(map(i->(2:i-1),Nu)) 
    @WaterLily.loop df[I,:] .= ∇²u(I,u)*nds(body,loc(0,I,T),t) over I ∈ inside(In)
    [sum(@inbounds(df[inside(In),i])) for i ∈ 1:N-1] |> Array
end
# 
function test()
    N=128; Λ=2
    body = AutoBody((x,t)->√sum(abs2,(x.-N÷2)./SA[1.,Λ])-N÷4/Λ)
    a = Simulation((N,N),(1.,0.),N;body); t = 0.0
    @WaterLily.loop a.flow.σ[I] = xnds(a.body,SA[N/2,N/2],loc(0,I),t,a.ϵ) over I ∈ inside(a.flow.p)
    flood(a.flow.σ;clims=(-1,1),legend=false)

    # # hydrostatic pressure force
    f1=[]; f2=[]; m=[]; resolutions = [16,32,64,128,256,512]
    for N ∈ resolutions
        Λ=1.0 # ellipse aspect ratio
        body = AutoBody((x,t)->√sum(abs2,(x.-N÷2)./SA[1.,Λ])-N÷4/Λ)
        a = Simulation((N,N),(1.,0.),N÷2;body,mem=Array,T=Float32)
        WaterLily.measure!(a.flow,a.body)
        @inside a.flow.p[I] = sdf(a.body,loc(0,I),0.0) >= 0 ? loc(0,I)[2] : 0
        force,moment = diagnostics(a,SA[N/2,N/2])
        push!(f1,-force/(π*(N÷4)^2)*Λ)
        push!(f2,∮nds_ϵ(a.flow.p,a.flow.f,a.body,0.0,0.0)/(π*(N÷4)^2)*Λ)
        push!(m,moment)
    end
    plot(title="Hydrostatic pressure force",xlabel="N",ylabel="force/πN²/Λ")
    plot!(resolutions,reduce(hcat,f1)[2,:],label="Diagnostic(sim)")
    plot!(resolutions,reduce(hcat,f2)[2,:],label="WaterLily.∮nds(p,body,t)")
    plot!(resolutions,m,label="Pressure moments")
    savefig("pressure_force_1.png")
end


"""Circle function"""
function circle(L=32;m=6,n=4,Re=80,U=1,T=Float32)
    radius, center = L/2, max(n*L/2,L)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((m*L,n*L), (U,0), radius; ν=U*radius/Re, body, T)
end
sim = circle(64;m=12,n=8,Re=80,U=1,T=Float64)

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1
forces_p = []; forces_ν = [];
forces_p_old = []; forces_ν2 = [];

# step and plot
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U

        # update flow
        mom_step!(sim.flow,sim.pois)
        
        # pressure force
        force = -2WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,0.0)
        push!(forces_p_old,force)
        force = -2∮nds_ϵ(sim.flow.p,sim.flow.f,sim.body,0.0)
        vforce = 2sim.flow.ν.*∮τnds(sim.flow.u,sim.flow.f,sim.body,0.0)
        vforce2 = 2sim.flow.ν.*∮τds(sim.flow.u,sim.flow.f,sim.body,0.0)
        push!(forces_p,force); push!(forces_ν,vforce); push!(forces_ν2,vforce2)
        # push!(p_trace,sim.flow.p[100,200])
        # update time
        t += sim.flow.Δt[end]
    end
  
    # print time step
    println("tU/L=",round(tᵢ,digits=4),",  Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ;
    @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(a[inside(a)],clims=(-10,10), legend=false); body_plot!(sim)
    contour!(sim.flow.p[inside(a)]',levels=range(-1,1,length=10),
             color=:black,linewidth=0.5,legend=false)
end
forces_p = reduce(vcat,forces_p')
forces_ν = reduce(vcat,forces_ν')
forces_ν2 = reduce(vcat,forces_ν2')
forces_p_old = reduce(vcat,forces_p_old')
# plot(forces_p[4:end,1]/(π*sim.L),label="pressure force")
# plot!(forces_p_old[4:end,1]/(π*sim.L),label="pressure force old")
plot(forces_ν[4:end,1]/(π*sim.L),label="viscous force")
plot!(forces_ν2[4:end,1]/(π*sim.L),label="viscous force2")
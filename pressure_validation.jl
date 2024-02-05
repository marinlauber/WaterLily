using WaterLily
using StaticArrays
using ForwardDiff
using LinearAlgebra: tr, norm, I # this might be an issue
using WaterLily: kern, ∂, inside_u

"""Circle function"""
function circle(L=32;m=6,n=4,Re=80,U=1,T=Float32)
    radius, center = L/2, max(n*L/2,L)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((m*L,n*L), (U,0), radius; ν=U*radius/Re, body, T)
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
    n*WaterLily.kern(clamp(d-ε,-1,1))/prod(1.0.+κ*d)
end
# stress tensor
∇²u(J::CartesianIndex{2},u) = @SMatrix [(1+I[i,j])*∂(i,j,J,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(J::CartesianIndex{3},u) = @SMatrix [(1+I[i,j])*∂(i,j,J,u) for i ∈ 1:3, j ∈ 1:3]
"""surface integral of the stress tensor"""
function ∮∇²u_nds(u::AbstractArray{T},df::AbstractArray{T},body::AutoBody,t=0) where {T}
    Nu,n = WaterLily.size_u(u); Inside = CartesianIndices(map(i->(2:i-1),Nu)) #df .= 0.0 
    @WaterLily.loop df[I,:] = ∇²u(I,u)*nds_ϵ(body,loc(0,I,T),t,1.0) over I ∈ Inside
    [sum(@inbounds(df[Inside,i])) for i ∈ 1:n] |> Array
end

include("examples/TwoD_plots.jl")

# # # hydrostatic pressure force
# f1=[]; f2=[]; f3=[]; resolutions = [16,32,64,128,256,512]
# for N ∈ resolutions
#     a = Flow((N,N),(1.,0.);f=Array,T=Float32)
#     sdf(x,t) = √sum(abs2,x.-N÷2)-N÷4
#     map(x,t) = x.-SVector(t,0)
#     body = AutoBody(sdf,map)
#     WaterLily.measure!(a,body)
#     @inside a.p[I] = sdf(loc(0,I),0) >= 0 ? loc(0,I)[2] : 0
#     push!(f1,WaterLily.∮nds(a.p,a.f,body,0.0)/(π*(N÷4)^2))
#     push!(f2,∮nds_ϵ(a.p,a.f,body,0.0,0.0)/(π*(N÷4)^2))
#     push!(f3,∮nds_ϵ(a.p,a.f,body,0.0)/(π*(N÷4)^2))
# end
# plot(title="Hydrostatic pressure force",xlabel="N",ylabel="force/πN²")
# plot!(resolutions,reduce(hcat,f1)[2,:],label="WaterLily.∮nds(p,body,t)")
# plot!(resolutions,reduce(hcat,f2)[2,:],label="curvature correction")
# plot!(resolutions,reduce(hcat,f3)[2,:],label="kernel at ϵ with curvature correction")
# savefig("pressure_force.png")

# function logger(fname::String="WaterLily")
#     ENV["JULIA_DEBUG"] = all
#     logger = FormatLogger(fname*".log"; append=false) do io, args
#         println(io, "@", args.level, " ", args.message)
#     end;
#     global_logger(logger);
# end

# logger("cylinder_force")

sim = circle(64;m=12,n=8,Re=80,U=1,T=Float64)

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1
forces_p = []; forces_ν = []; p_trace = [];
forces_p_old = []

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
        vforce = 2sim.flow.ν.*∮∇²u_nds(sim.flow.u,sim.flow.f,sim.body,0.0)
        push!(forces_p,force); push!(forces_ν,vforce)
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
    # flood(sim.flow.p[inside(a)],clims=(-1,1)); body_plot!(sim)
    # plot!([100],[200],marker=:o,color=:red,markersize=2,legend=false)
end
forces_p = reduce(vcat,forces_p')
forces_ν = reduce(vcat,forces_ν')
forces_p_old = reduce(vcat,forces_p_old')
plot(forces_p[4:end,1]/(π*sim.L),label="pressure force")
plot!(forces_p_old[4:end,1]/(π*sim.L),label="pressure force old")
plot!(forces_ν[4:end,1]/(π*sim.L),label="viscous force")
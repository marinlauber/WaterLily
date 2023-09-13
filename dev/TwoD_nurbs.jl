using WaterLily
using StaticArrays
using ForwardDiff
using LinearAlgebra: norm
T = Float64
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

# # make sim with wired spline
# p=3
# Points = 3L.+0.5*L*Array(reduce(hcat,[SVector(i-5, 3sin(i^2)) for i in 1:9])')
# Points = Array(Points')
# Body = SplineBody(Points;deg=p,thk=thk,compose=false,T=Float64)
# sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)
# flood(sim.flow.μ₀[:,:,1])

# circle
degP = 2
Points = hcat([0.,-1.],[1.,-1.],[ 1.,0.],[1.,1.],
              [0.,1. ],[-1.,1.],[-1.,0.],[-1.,-1.],
              [0.,-1.])*L .+ 3L
weights = [1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots = [0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1]

Body = SplineBody(Points,weights,knots;deg=degP,thk=thk,compose=false,T=Float64)
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,ϵ,T=Float64)
# flood(sim.flow.μ₀[:,:,1])

# Define B-spline
Xs = WaterLily.evaluate_nurbs(Points, weights, knots, 0:0.01:1, degP)
xs = Xs[1,:]; ys=Xs[2,:]

t₀ = round(sim_time(sim))
duration = 15; tstep = 0.1; force = []
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
    Plots.plot!(xs,ys,color=:black,lw=4,legend=false)
    Plots.plot!(Points[1,:],Points[2,:],markers=:o,legend=false)

    # compute forces 
    fi = sum(WaterLily.∮ds(sim.flow.p,Body,collect(0:1/2L:1)),dims=2)
    push!(force,fi)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# forces = reduce(hcat,force)
# # gif(anim, "circle_NURBS_flow.gif", fps=24)s

# hydrostatic pressure with hollow body
function pλ(x, center, radius)
    d = √sum(abs2, x .- [center,center]) - radius
    if d≤0
        return 0.0
    end
    return x[2]
end

# pressure force
apply!(x->pλ(x,3L,L),sim.flow.p)
force = WaterLily.∮ds(sim.flow.p,Body,collect(0:1/2L:1))
display(sum(force,dims=2)/(π*L^2))
flood(sim.flow.p)

# Plots.plot!(xs,ys,color=:black,lw=4,legend=false)
# Plots.plot!(Points[1,:],Points[2,:],markers=:o,legend=false)
# # test
# ts=collect(0:1/2L:1)
# x = WaterLily.evaluate_nurbs(Points, weights, knots, ts, Body.deg)
# n = [ForwardDiff.derivative(t->WaterLily.nurbs(Points, weights, knots, t, d=degP), t) for t in ts]
# n = reduce(hcat,n)
# δd = Float32(1.0) # we can intepolate on the surface itself
# f = zeros(T,size(x,2)-1,2)
# for i in 1:size(x,2)-1  
#     dx,dy = x[1,i+1]-x[1,i],x[2,i+1]-x[2,i]
#     dl = √(dx^2+dy^2)
#     xi = SVector( (x[1,i+1]+x[1,i])/2., (x[2,i+1]+x[2,i])/2.)
#     ni = SVector(-(n[2,i+1]+n[2,i])/2., (n[1,i+1]+n[1,i])/2.)
#     ni = ni/norm(ni)
#     p_at_X = WaterLily.interp(convert.(T,xi.+δd.*ni),sim.flow.p)
#     p_at_X -= WaterLily.interp(convert.(T,xi.-δd.*ni),sim.flow.p)
#     f[i,:] = p_at_X.*dl.*ni
#     Plots.plot!([xi[1]],[xi[2]],markershape=:circle,markersize=2,markercolor=:red)
#     # Plots.plot!([xi[1],xi[1]+δd*ni[1]],[xi[2],xi[2]+δd*ni[2]],markershape=:circle,markersize=2,markercolor=:blue)
#     Plots.plot!([xi[1],xi[1]-δd*ni[1]],[xi[2],xi[2]-δd*ni[2]],markershape=:circle,markersize=2,markercolor=:blue)
# end
# display(Plots.plot!())
# display(sum(f,dims=1)/(π*(L)^2))
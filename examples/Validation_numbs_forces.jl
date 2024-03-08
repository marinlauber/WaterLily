using WaterLily
using ParametricBodies
using StaticArrays
using Plots
include("TwoD_plots.jl")

# parameters
L=2^6
Re=80
U=1
U=1
ϵ=0.5
thk=2ϵ+√2
center = SA[4L,4L]
radius = L/2

# NURBS points, weights and knot vector for a circle
cps = SA[1 1 0 -1 -1 -1  0  1 1
         0 1 1  1  0 -1 -1 -1 0]*radius .+ center
weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

# make a nurbs curve
circle = NurbsCurve(cps,knots,weights)

# make a body and a simulation
Body = ParametricBody(circle,(0,1))
sim = Simulation((12L,8L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64,mem=Array)

function pλ(x)
    if √sum(abs2,x.-center)<=radius
        return 0.0
    else
        return x[2]
    end
end

# perimenter of the circle
len = integrate(circle;N=128)
println("Exact :$(L/2) , Numerical: ", len/2π)

# uniform pressure field
WaterLily.apply!(pλ,sim.flow.p)
println(pforce(Body.surf,sim.flow.p;N=32)/(π*(radius)^2))
flood(sim.flow.p,shift=[1.5,1.5])
plot!(Body.surf)

# reset pressure
sim.flow.p.=0.0

# intialize
t₀ = sim_time(sim)
duration = 200.
tstep = 0.2
vforces = []; pforces = []
# run
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
        # make pressure inside body exactly zero
        @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0.0),ϵ)*sim.flow.p[I]
        pforce = ParametricBodies.pforce(sim.body.surf,sim.flow.p;N=64)
        vforce = ParametricBodies.vforce(sim.body.surf,sim.flow.u;N=64)
        push!(pforces,pforce[1]); push!(vforces,vforce[1])
    end

    # flood plot
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    contourf(clamp.(sim.flow.σ,-10,10)'|>Array,dpi=300,
             color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)
    plot!(Body.surf; add_cp=true)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "cylinder_flow_nurbs.gif", fps=24)
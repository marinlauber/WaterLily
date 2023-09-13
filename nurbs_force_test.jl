using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using Splines
using ForwardDiff
include("examples/TwoD_plots.jl")

# parameters
L=2^5
Re=250
U =1
ϵ=0.5
thk=2ϵ+√2
center = SA[L,L]
radius = L/2

# NURBS points, weights and knot vector for a circle
cps = SA[1 1 0 -1 -1 -1  0  1 1
         0 1 1  1  0 -1 -1 -1 0]*radius .+ center
weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

# make a nurbs curve
circle = NurbsCurve(MMatrix(cps),knots,weights)

# make a body and a simulation
Body = DynamicBody(circle,(0,1))
sim = Simulation((2L,2L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

function pλ(x)
    if √sum(abs2,x.-center)<=radius
        return 0.0
    else
        return x[2]
    end
end

# length of the beam
len = integrate(circle;N=L)
println(len)

# uniform pressure field
WaterLily.applyS!(pλ,sim.flow.p)
println(force(Body.surf,sim.flow.p;N=128)/(π*(radius)^2))
flood(sim.flow.p,shift=[0.5,0.5])
plot!(Body.surf)

# # intialize
# t₀ = sim_time(sim)
# duration = 10
# tstep = 0.1

# # run
# anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

#     # update until time tᵢ in the background
#     t = sum(sim.flow.Δt[1:end-1])
#     while t < tᵢ*sim.L/sim.U
#         # # random update
#         # new_pnts = SA[-1     0   1
#         #               0.5 0.25+0.5*sin(π/4*t/sim.L) 0]*L .+ [2L,3L]
#         # ParametricBodies.update!(Body,new_pnts,sim.flow.Δt[end])
#         # measure!(sim,t)
#         mom_step!(sim.flow,sim.pois) # evolve Flow
#         t += sim.flow.Δt[end]
#     end

#     # flood plot
#     @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
#     contourf(clamp.(sim.flow.σ,-10,10)',dpi=300,
#              color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
#              aspect_ratio=:equal, legend=false, border=:none)
#     plot!(Body.surf; add_cp=true)

#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
# # save gif
# # gif(anim, "DynamicBody_flow.gif", fps=24)

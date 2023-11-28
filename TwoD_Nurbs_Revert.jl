using WaterLily
using ParametricBodies
using StaticArrays
include("examples/TwoD_plots.jl")

# parameters
L=2^4
Re=250
U =1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# define a flat plat at and angle of attack
cps = SA[-1   0   1
         0.5 0.25 0]*L .+ [2L,3L]

# needed if control points are moved
cps_m = MMatrix(cps)
weights = SA[1.,1.,1.]
knots =   SA[0,0,0,1,1,1.]

# make a nurbs curve
# circle = NurbsCurve(cps_m,knots,weights)
circle = BSplineCurve(cps_m;degree=2)

# make a body and a simulation
Body = DynamicBody(circle,(0,1))
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# duration of the simulation
duration = 5
step = 0.1
t₀ = 0.0

@time @gif for tᵢ in range(t₀,t₀+duration;step)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])

    while t < tᵢ*sim.L/sim.U
        
        println("tᵢ=$tᵢ, t=$t, Δt=$(sim.flow.Δt[end])")

        # save at start of iterations
        WaterLily.store!(sim.flow); iter=1; r₂ = 1.0;
        ParametricBodies.store!(sim.body);

        # iterative loop
        while r₂ > 1e-3
            println("residuals: ",rand()/iter^2)
            # update flow
            # random update
            new_pnts = SA[-1     0   1
                          0.5 0.25+0.5*sin(π/4*t/sim.L) 0]*L .+ [2L,3L]
            ParametricBodies.update!(sim.body,new_pnts,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            if iter>3 # if we converge, we exit to avoid reverting the flow
                println("Converged...")
                break
            end 

            # if we have not converged, we must revert
            WaterLily.revert!(sim.flow)
            ParametricBodies.revert!(sim.body);
            iter += 1
        end

        # finish the time step
        Δt = sim.flow.Δt[end]
        t += Δt
    end
    # if tᵢ==5.0
    #     WaterLily.store!(sim.flow)
    # end
    # if tᵢ==10.0
    #     WaterLily.revert!(sim.flow)
    # end

    # plot vorticity
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ; shift=(-0.5,-0.5), clims=(-5,5))
    plot!(sim.body.surf)
    plot!(title="tU/L $tᵢ")
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end

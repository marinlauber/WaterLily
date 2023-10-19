using WaterLily
include("examples/TwoD_plots.jl")

# some definitons
U = 1
Re = 250
m, n = 2^6, 2^7
n = n*2
m = m*2
println("$m x $n: ", prod((m,n)))
radius, center = 16, 64

# make a circle body
body = AutoBody((x,t)->√sum(abs2, x .- [center,center]) - radius)
z = [radius*exp(im*θ) for θ ∈ range(0,2π,length=33)]

# make a simulation
sim = Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, T=Float64)

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

        # iterative loop
        while r₂ > 1e-3
            println("residuals: ",rand()/iter^2)
            # update flow
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            if iter>3 # if we converge, we exit to avoid reverting the flow
                println("Converged...")
                break
            end 

            # if we have not converged, we must revert
            WaterLily.revert!(sim.flow)
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
    plot!(title="tU/L $tᵢ")
    body_plot!(sim)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end

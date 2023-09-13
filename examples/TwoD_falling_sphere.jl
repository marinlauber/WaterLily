using WaterLily
using Plots; gr()
using StaticArrays
include("TwoD_plots.jl")
let
    # parameters
    Re = 250
    U = 1
    L = 2^5
    radius, center = L/2, SA[3L,5L]
    duration = 10
    step = 0.1

    # fsi parameters
    mₐ = π*radius^2 # added-mass coefficent circle
    m = 1mₐ
    vel = [0.,0.]
    a0 = [0.,0.]
    pos = [0.,0.]
    g = [0.,-9.81/100]
    t_init = 0

    # motion definition
    function map(x,t)
        x - pos - (t-t_init).*vel
    end

    # make a body
    circle = AutoBody((x,t)->√sum(abs2, x .- center) - radius, map)

    # generate sim
    sim = Simulation((6L,6L), (0,0), radius; ν=U*radius/Re, U, body=circle)

    # get start time
    t₀ = round(sim_time(sim))

    @time @gif for tᵢ in range(t₀,t₀+duration;step)

        # update
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U

            # measure body
            measure!(sim,t)

            # update flow
            mom_step!(sim.flow,sim.pois); sim.flow.Δt[end] = min(0.25,sim.flow.Δt[end])
            
            # pressure force
            force = -WaterLily.∮nds(sim.flow.p,sim.flow.f,circle,t)
            # @show force
            # # compute motion and acceleration 1DOF
            Δt = sim.flow.Δt[end]
            accel = (force + m.*g + mₐ.*a0)/(m + mₐ)
            # @show accel
            pos .+= Δt.*(vel+Δt.*accel./2.) 
            vel .+= Δt.*accel
            a0 .= accel
            
            # update time, must be done globaly to set the pos/vel correctly
            t_init = t; t += Δt
        end

        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ; shift=(-0.5,-0.5),clims=(-10,10))
        body_plot!(sim)
        
        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
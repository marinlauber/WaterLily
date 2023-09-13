using WaterLily
using LoggingExtras
# using RecipesBase: @recipe, @series

# @recipe function f(ml::MultiLevelPoisson)
#     seriestype := :path
#     primary := false
#     n = reshape(ml.n,(2,length(ml.n)÷2))
#     @series begin
#         linecolor := :blue
#         n[1,:]
#     end
#     @series begin
#         linecolor := :orange
#         n[2,:]
#     end
# end

# function Logging.handle_message(logger::SimpleLogger,
#                                 lvl, msg, _mod, group, id, file, line;
#                                 kwargs...)
#     # Write the formatted log message to logger.io
#     println(logger.io, "[", lvl, "] ", msg)
# end

function logger(fname::String="WaterLily")
    ENV["JULIA_DEBUG"] = all
    logger = FormatLogger(fname*".log"; append=false) do io, args
        println(io, "@", args.level, " ", args.message)
    end;
    global_logger(logger);
end

include("examples/TwoD_plots.jl")
function circle(n,m;radius=m/16,center=m/2,Re=250,U=1)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, T=Float64, exitBC=false)
end

# for N in [2^6,2^7,2^8]
#     sim = circle(3*N,2*N)
#     for i in 1:100
#         mom_step!(sim.flow,sim.pois)
#     end
#     n = reshape(sim.pois.n,(2,length(sim.pois.n)÷2))
#     plot(n[1,:],label="Predictor MG iter", title="MG iter 2D circle, N=$N", ylims=(0,32));
#     plot!(n[2,:],label="Corrector MG iter")
#     savefig("MG_iter_2D_circle_$(N)_tol.png")
# end

for N in [2^6,2^7,2^8]
    fname = "pressure_debug/MG_iter_2D_circle_$(N)_Vcycle_pop"
    sim = circle(3*N,2*N)

    # to log iterations
    logger(fname)

    # utils
    R = inside(sim.flow.p)
    anim = @animate for i in 1:N
        mom_step!(sim.flow,sim.pois)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R]; clims=(-5,5), legend=false); body_plot!(sim)
        contour!(sim.flow.p[R]',levels=range(-1,1,length=10),color=:black,
                linewidth=0.5,legend=false,title="iter = $i")
        println("iter=",i,", Δt=",round(sim.flow.Δt[end],digits=3)," mglevels=",length(sim.pois.levels))
    end
    gif(anim,fname*".gif")
    n = reshape(sim.pois.n,(2,length(sim.pois.n)÷2))
    plot(n[1,:],label="Predictor MG iter", title="MG iter 2D circle, N=$N", ylims=(0,32));
    plot!(n[2,:],label="Corrector MG iter")
    savefig(fname*".png")
end
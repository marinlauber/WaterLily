using WaterLily
using ParametricBodies
using JLD2
using Plots
include("TwoD_plots.jl")

data = jldopen("fillament_5.jld2","r")
L = 2^5
xlim,ylim = data["frame_$(length(data))"]["X"]
pos_x = []
pos_y = []
velocity = []
x0 = sum(data["frame_1"]["pnts"];dims=2)/length(data["frame_1"]["pnts"])/L
@gif for i in 1:length(data)
    # get data
    frame = data["frame_$i"]
    mod(i,10)==0 && @show i, frame["U"]
    ω = frame["ω"]
    X = frame["X"]/L
    pnts = frame["pnts"]/L
    nurbs = BSplineCurve(pnts;degree=3)
    t = frame["t"]
    N = size(ω)

    push!(pos_x, pnts[1,[1,end]].+X[1])
    push!(pos_y, pnts[2,[1,end]].+X[2])
    push!(velocity, norm(frame["U"]))

    # plot the trajectory and velocity
    plot(getindex.(pos_x,1),getindex.(pos_y,1),linez=velocity,label=:none,colorbar=true)
    plot!(getindex.(pos_x,2),getindex.(pos_y,2),linez=velocity,label=:none,colorbar=true)

    # plot the vorticity and the domain
    clims=(-10,10)
    contourf!(axes(ω,1)/L.+X[1],axes(ω,2)/L.+X[2],
              clamp.(ω',clims[1],clims[2]),linewidth=0,levels=10,color=palette(RdBu_alpha,256),
              clims=clims,aspect_ratio=:equal;dpi=300)
    
    # plot the spline
    plot!(nurbs;add_cp=false,shift=(X[1],X[2]))
    
    # plot limits
    xlims!(-N[1]/L,(N[1]+xlim)/L); ylims!(ylim/L,N[2]/L)
    xlabel!("x/L"); ylabel!("y/L")
    plot!(title="tU/L $(round(t,digits=2))",aspect_ratio=:equal)
end
using WaterLily
using StaticArrays
include("examples/TwoD_plots.jl")
# include("nurbs_sdf.jl")
# export bernstein_B,bernstein_dB

# struct SplineBody<:AbstractBody
#     pts :: AbstractArray{Float64,2}
#     deg :: Integer
#     sdf :: Function
#     ths :: Float64
#     function SplineBody(pts;deg=3,thk=1.)
#         sdf(x,t) = nurbs_sdf(pts;deg=deg)([x...]) .- thk/2.
#         new(pts,deg,sdf,thk)
#     end
# end

# function nurbs_sdf(Points; deg=3)
#     # Define cubic Bezier
#     curve(t) = [(bernstein_B(t, deg) * Points)...]
#     dcurve(t) =[(bernstein_dB(t, deg)* Points)...]

#     # Signed distance to the curve
#     candidates(X) = union(-1,1,find_zeros(t -> (X-curve(t))'*dcurve(t),-1,1,naive=true,no_pts=3,xatol=0.01))
#     function distance(X,t)
#         V = X-curve(t)
#         √(V'*V)
#     end
#     X -> argmin(abs, distance(X,t) for t in candidates(X))
# end

# parameters
L=2^5
Re=250
U=1;amp=π/4
ϵ=0.5
thk=2ϵ+√2

# Line segment S
# function sdf(x,t)
#     y = x .- SA[0,clamp(x[2],-L/2,L/2)]
#     √sum(abs2,y)-thk/2
# end
# function sdf(x,t)
#     d = nurbs_sdf(Array([[0,-L/2] [0,-L/4] [0.,L/4] [0.,L/2]]');deg=3)([x[1],x[2]])
#     return d - thk/2
# end

# Oscillating motion and rotation
function map(x,t)
    α = amp*cos(t*U/L); R = SA[cos(α) sin(α); -sin(α) cos(α)]
    R * (x - SA[3L-L*sin(t*U/L),4L])
end

# make sim
# Points = Array([[3L,L] [3L,4L/3] [3L,5L/3] [3L,2L]]')
Points = Array([[0,-L/2] [0,-L/4] [0.,L/4] [0.,L/2]]')
Body = SplineBody(Points;map=map,deg=3,thk=thk)
sim = Simulation((6L,6L),(0,0),L;U,ν=U*L/Re,body=Body,ϵ)

# @time sim_step!(sim,0.1)
a = sim.flow.σ;
@inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
flood(a[inside(a)],clims=(-5,5))
body_plot!(sim)

force = WaterLily.∮ds(Body,sim.flow.p)
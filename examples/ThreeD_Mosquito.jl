using WaterLily
using StaticArrays
using CSV
using DataFrames
using Interpolations
include("ThreeD_Plots.jl")

function geom!(d,sim,t)
    a = sim.flow.σ
    WaterLily.measure_sdf!(a,sim.body,t)
    copyto!(d,a[inside(a)]) # copy to CPU
end

# very simple interpolation function
interp(x,y) = interpolate((vcat(0.0,x),), vcat(y[end],y), Gridded(Linear()))

# read the CSV data
df = CSV.read("kinematicdata_Aaegypti_norm.csv", DataFrame;
               header=["t","ϕ","θ","α"])

# # plot it to check
# plot(df[!,:t], df[!,:ϕ], label="ϕ")
# plot!(df[!,:t], df[!,:θ], label="θ")
# plot!(df[!,:t], df[!,:α], label="α")

# generate interpolation functions
ϕ = interp(df[!,:t], df[!,:ϕ]); 
@assert ϕ(0)==ϕ(1)
@assert all(ϕ(df[!,:t])≈df[!,:ϕ]) # check that we interpolate correclty
θ = interp(df[!,:t], df[!,:θ]);
α = interp(df[!,:t], df[!,:α]);

# plot again but check periodicity
# plot!(0:0.01:2, ϕ(mod.(0:0.01:2,1)), style=:dash, label="ϕ interp")
# plot!(0:0.01:2, θ(mod.(0:0.01:2,1)), style=:dash, label="θ interp")
# plot!(0:0.01:2, α(mod.(0:0.01:2,1)), style=:dash, label="α interp")

# define mesh
L = 2^5
U = 1
Re = 120
ν = U*L/Re
center, radius = SA[L,L], L/4
ϵ = 0.5
thk = 2ϵ+√3
AR = 3


# the mapping
function map(x,t)
    # apply rotation for pitch
    _α = π/2 - α(t) # positive pitch increase AoA
    _θ = θ(t)  # positive upward
    _ϕ = ϕ(t)      # positive to the rear of the mosquito
    # rotation mmatrix
    Rx = SA[1 0 0; 0 cos(_θ) -sin(_θ);0 sin(_θ) cos(_θ)] # theta    
    Ry = SA[cos(_α) 0 sin(_α); 0 1 0; -sin(_α) 0 cos(_α)] # alpha
    Rz = SA[cos(_ϕ) -sin(_ϕ) 0; sin(_ϕ) cos(_ϕ) 0; 0 0 1] # phi
    return Ry*Rz*Rx*(x .- SA[L,0,L])
end

# define a body from an plate with elipsoidal cross section
elipse = AutoBody((x,t)->√sum(abs2,SA[x[1],(x[2]-L)/AR,])-radius,map)
upper_lower = AutoBody((x,t)->-(abs(x[3])-thk/2),map)
body = elipse-upper_lower

# body = AutoBody(sdf,map)
sim = Simulation((2L,2L,2L),(0,0,0),L;ν,body,T=Float32)

# Set up geometry viz
d = similar(sim.flow.σ,size(inside(sim.flow.σ))) |> Array
geom = geom!(d,sim,0.0) |> Observable;
fig, _, _ = contour(geom, levels=[0], alpha=1)

len = 16
# record(fig,"mosquito.mp4",1:len) do frame
foreach(1:len) do frame
    @show frame
    geom[] = geom!(d,sim,frame/len);
end

# function jelly(p=5;Re=5e2,mem=Array,U=1)
#     # Define simulation size, geometry dimensions, & viscosity
#     n = 2^p; R = 2n/3; h = 4n-2R; ν = U*R/Re

#     # Motion functions
#     ω = 2U/R
#     @fastmath @inline A(t) = 1 .- SA[1,1,0]*0.1*cos(ω*t)
#     @fastmath @inline B(t) = SA[0,0,1]*((cos(ω*t)-1)*R/4-h)
#     @fastmath @inline C(t) = SA[0,0,1]*sin(ω*t)*R/4

#     # Build jelly from a mapped sphere and plane
#     sphere = AutoBody((x,t)->abs(√sum(abs2,x)-R)-1, # sdf
#                       (x,t)->A(t).*x+B(t)+C(t))     # map
#     plane = AutoBody((x,t)->x[3]-h,(x,t)->x+C(t))
#     body =  sphere-plane

#     # Return initialized simulation
#     Simulation((n,n,4n),(0,0,-U),R;ν,body,mem,T=Float32)
# end

# function geom!(md,d,sim,t=WaterLily.time(sim))
#     a = sim.flow.σ
#     WaterLily.measure_sdf!(a,sim.body,t)
#     copyto!(d,a[inside(a)]) # copy to CPU
#     mirrorto!(md,d)         # mirror quadrant
# end

# function ω!(md,d,sim)
#     a,dt = sim.flow.σ,sim.L/sim.U
#     @inside a[I] = WaterLily.ω_mag(I,sim.flow.u)*dt
#     copyto!(d,a[inside(a)]) # copy to CPU
#     mirrorto!(md,d)         # mirror quadrant
# end

# function mirrorto!(a,b)
#     n = size(b,1)
#     a[reverse(1:n),reverse(1:n),:].=b
#     a[reverse(n+1:2n),1:n,:].=a[1:n,1:n,:]
#     a[:,reverse(n+1:2n),:].=a[:,1:n,:]
#     return a
# end

# import CUDA
# using GLMakie
# begin
#     # Define geometry and motion on GPU
#     sim = jelly(mem=CUDA.CuArray);

#     # Create CPU buffer arrays for geometry flow viz 
#     a = sim.flow.σ
#     d = similar(a,size(inside(a))) |> Array; # one quadrant
#     md = similar(d,(2,2,1).*size(d))  # hold mirrored data

#     # Set up geometry viz
#     geom = geom!(md,d,sim) |> Observable;
#     fig, _, _ = contour(geom, levels=[0], alpha=0.01)

#     #Set up flow viz
#     ω = ω!(md,d,sim) |> Observable;
#     volume!(ω, algorithm=:mip, colormap=:algae, colorrange=(1,10))
#     fig
# end

# # Loop in time
# # record(fig,"jelly.mp4",1:200) do frame
# foreach(1:100) do frame
#     @show frame
#     sim_step!(sim,sim_time(sim)+0.05);
#     geom[] = geom!(md,d,sim);
#     ω[] = ω!(md,d,sim);
# end
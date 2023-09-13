using WaterLily
using StaticArrays
using ParametricBodies
# struct PlanarParametricBody <: AbstractParametricBody
#     surf::S    #ξ = surf(uv,t)
#     locate::L  #uv = locate(ξ,t)
#     map::M     #ξ = map(x,t)
# end
# function PlanarParametricBody(surf,locate;map=(x,t)->x,T=Float64)
    
#     PlanarParametricBody(surf,locate,map)
# end
# function measure(body::PlanarParametricBody,x,t)
#     # Surf props and velocity in ξ-frame
#     d,n,uv = surf_props(body,x,t)
#     dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(x,t),t)

#     # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
#     dξdx = ForwardDiff.jacobian(x->body.map(x,t),x)
#     return (d,dξdx\n/body.scale,dξdx\dξdt)
# end
# function surf_props(body::PlanarParametricBody,x,t)
#     # Map x to ξ and locate nearest uv
#     ξ = body.map(x,t)
#     uv = body.locate(ξ,t)

#     # Get normal direction and vector from surf to ξ
#     n = norm_dir(body.surf,uv,t)
#     p = ξ-body.surf(uv,t)

#     # Fix direction for C⁰ points, normalize, and get distance
#     notC¹(body.locate,uv) && p'*p>0 && (n = p)
#     n /=  √(n'*n)
#     return (body.scale*dis(p,n),n,uv)
# end

function circle(p=4;Re=250,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    L=2^p
    center,r = SA[3L,3L], L/2
    ν = U*L/Re

    # define a spline distance function and move to center
    cps = SA[4 4 0 -4 -4 -4  0  4 4
             0 1 1  1  0 -1 -1 -1 0].*L/4 .+ center
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
    
    circle = NurbsCurve(cps,knots,weights)
    planar_ellipse = ParametricBody(circle,(0,1))

    # nurbs for the time series
    cps = SA[0 1 2 3 4
             0 1 0 -1 0]
    time_curve = BSplineCurve(cps; degree=3)

    # cylinder SDF
    function ellipse(x,t)
        x = x .- SA[3L,3L,L]
        r_ϕ = √(x[2]^2+x[3]^2) # map into cylindrical coordinate system
        √sum(abs2,SA[x[1],r_ϕ])-r
    end
    # ellipse(x,t) = ParametricBodies.sdf(planar_ellipse,SA[x[1],x[2]],0.0)
    # map(x,t) = x .+ SA[0.,0.0*time_curve(mod(t/L,1),0.0)[2],0.0]
    map(x,t) = x .+ SA[0.,L/2*sin(π*t/L),0.]
    # make a body
    body = AutoBody(ellipse,map;compose=true)

    Simulation((8L,6L,2L),(U,0,0),L;ν,body,mem)
end

# import CUDA
# @assert CUDA.functional()
sim = circle();

# make the writer
wr = vtkWriter("TwoD_circle")

# intialize
t₀ = sim_time(sim)
duration = 0.1
tstep = 0.1

# step and write
@time for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)
    write!(wr,sim)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)

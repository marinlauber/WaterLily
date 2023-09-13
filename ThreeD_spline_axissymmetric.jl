using WaterLily
using StaticArrays
using ParametricBodies

function make_sim(p=4;Re=250,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    L=2^p
    center,r = SA[3L,3L,L], L/2
    ν = U*L/Re

    # define a spline distance function
    cps = SA[4 4 0 -4 -4 -4  0  4 4
             0 1 1  1  0 -1 -1 -1 0].*L/4
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
    
    circle = NurbsCurve(cps,knots,weights)
    planar_ellipse = ParametricBody(circle,(0,1))

    function ellipse(x,t)
        x = x .- center
        r_ϕ = √(x[2]^2+x[3]^2) # map into cylindrical coordinate system
        # √sum(abs2,SA[x[1],r_ϕ])-r # standard signed sistance function for a sphere
        ParametricBodies.sdf(planar_ellipse,SA[x[1],r_ϕ],0.0)
    end
    
    # make a body
    body = AutoBody(ellipse)

    Simulation((8L,6L,2L),(U,0,0),L;ν,body,mem)
end

# import CUDA
# @assert CUDA.functional()
sim = make_sim();

# make the writer
wr = vtkWriter("ThreeD_nurbs")

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

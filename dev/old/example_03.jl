function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(clamp.(ω,-limit,limit)',dpi=300,
    color=palette(:RdBu_11), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)

function coxDeBoor(knots, u, k, d, count)
    """
        coxDeBoor(knots, u, k, d, count)

    Compute the Cox-De Boor recursion for B-spline basis functions.

    The `coxDeBoor` function computes the Cox-De Boor recursion for B-spline basis functions,
    used in the evaluation of B-spline curves and surfaces.

    Arguments:
    - `knots`: An array of knot values.
    - `u`: The parameter value at which to evaluate the B-spline basis function.
    - `k`: The index of the current knot interval.
    - `d`: The degree of the B-spline basis function.
    - `count`: The number of control points.

    Returns:
    The value of the B-spline basis function at parameter `u` and knot interval `k`.
    """
    if (d == 0)
        return Int(((knots[k+1] <= u) && (u < knots[k+2])) || ((u >= (1.0-1e-12)) && (k == (count-1))))
    end
    return (((u-knots[k+1])/max(√eps(u), knots[k+d+1]-knots[k+1]))*coxDeBoor(knots, u, k, (d-1), count)
          + ((knots[k+d+2]-u)/max(√eps(u), knots[k+d+2]-knots[k+2]))*coxDeBoor(knots, u, (k+1), (d-1), count))
end

"""
    BSplineCurve(cps; degree=3, mem=Array)

Define a B-spline curve.
- `cps`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
- `mem`: Array memory type
"""
struct BSplineCurve{A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    cps::A
    knots::V
    weights::W
    count::Int
    degree::Int
end
function BSplineCurve(cps;degree=3) 
    count,T = size(cps, 2),promote_type(eltype(cps),Float32)
    knots = [zeros(T,degree); collect(T,range(0, count-degree) / (count-degree)); ones(T,degree)]
    wgts = ones(T,count)
    BSplineCurve(cps,SA[knots...],SA[wgts...],count,degree)
end
function NURBSCurve(cps,wgts,knot)
    count,T = size(cps, 2),promote_type(eltype(cps),Float32)
    degree = length(knot)-count-1
    BSplineCurve(cps,SA{T}[knots...],SA{T}[wgts...],count,degree)
end
function (l::BSplineCurve)(s::T,t) where {T} # `t` is currently unused
    @assert 0 ≤ s ≤ 1 "Parameter `s` must be in the range [0,1]"
    pt = zero(l.cps[:,1]); wsum=T(0.0)
    for k in range(0, l.count-1)
        prod = coxDeBoor(l.knots, s, k, l.degree, l.count) * l.weights[k+1]
        pt += prod * l.cps[:, k+1]
        wsum += prod
    end
    return pt / wsum
end

# Define a circle using degree=2 NURBS.
using StaticArrays
cps = SA[5 5 0 -5 -5 -5  0  5 5
         0 5 5  5  0 -5 -5 -5 0]
weights = [1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots = [0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
circle = NURBSCurve(cps,weights,knots)

# Check bspline is type stable. Note that using Float64 for cps will break this!
@assert isa(circle(0.,0),SVector)
@assert all([eltype(circle(zero(T),0))==T for T in (Float32,Float64)])

# Create curve and check winding direction
using LinearAlgebra
cross(a,b) = det([a;;b])
@assert all([cross(circle(s,0.),circle(s+0.1,0.))>0 for s in range(.0,.9,10)])

# Check NURBS is type stable. Note that using Float64 for cps will break this!
@assert isa(circle(0.,0),SVector)
@assert all([eltype(circle(zero(T),0))==T for T in (Float32,Float64)])

# Wrap the shape function inside the parametric body class and check measurements
using Pkg
Pkg.activate("/home/marin/Workspace/ParametricBodies.jl")
using ParametricBodies
body = ParametricBody(circle, (0,1));
@assert all(measure(body,[-6,0],0) .≈ [1,[-1,0],[0,0]])
@assert all(measure(body,[ 5,5],0) .≈ [√(5^2+5^2)-5,[√2/2,√2/2],[0,0]])
@assert all(measure(body,[-5,5],0) .≈ [√(5^2+5^2)-5,[-√2/2,√2/2],[0,0]])

# test on CUDA
# using CUDA; @assert CUDA.functional()
# CUDA.@allowscalar @assert all(measure(body,[-6,0],0) .≈ [1,[-1,0],[0,0]])
# CUDA.@allowscalar @assert all(measure(body,[ 5,5],0) .≈ [√(5^2+5^2)-5,[√2/2,√2/2],[0,0]])
# CUDA.@allowscalar @assert all(measure(body,[-5,5],0) .≈ [√(5^2+5^2)-5,[-√2/2,√2/2],[0,0]])

# make a flow and test
using WaterLily
cps = SA[80  80  64  48 48 48 64 80 80
         96 112 112 112 96 80 80 80 96]
circle = NURBSCurve(cps,weights,knots)
Body = ParametricBody(circle, (0,1));
L = 2^5; U=1; Re=250;
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body)

t₀ = round(sim_time(sim))
duration = 15; tstep = 0.1; force = []
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
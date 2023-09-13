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

# Define square using degree=1 BSpline.
using StaticArrays
cps = SA[5 5 0 -5 -5 -5  0  5 5
         0 5 5  5  0 -5 -5 -5 0]
square = BSplineCurve(cps,degree=1)

# Check bspline is type stable. Note that using Float64 for cps will break this!
@assert isa(square(0.,0),SVector)
@assert all([eltype(square(zero(T),0))==T for T in (Float32,Float64)])

# Create curve and check winding direction
using LinearAlgebra
cross(a,b) = det([a;;b])
@assert all([cross(square(s,0.),square(s+0.1,0.))>0 for s in range(.0,.9,10)])

# Wrap the shape function inside the parametric body class and check measurements
using Pkg
Pkg.activate("/home/marin/Workspace/ParametricBodies.jl")
using ParametricBodies
body = ParametricBody(square, (0,1));
@assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
@assert all(measure(body,[8,2],0) .≈ [ 3,[1,0],[0,0]])


## circle
deg = 2
cps = SA[5 5 0 -5 -5 -5  0  5 5
         0 5 5  5  0 -5 -5 -5 0]
weights = [1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots = [0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1]
circle = NURBSCurve(cps,weights,knots)
body = ParametricBody(circle, (0,1));

# Check NURBS is type stable. Note that using Float64 for cps will break this!
@assert isa(circle(0.,0),SVector)
@assert all([eltype(circle(zero(T),0))==T for T in (Float32,Float64)])

# test actual distance function
using Plots
# range and storage
x = range(-10, 10, 20)
z = zeros(length(x),length(x))
sdf(x,y) = measure(body,[x,y],0)[1]
for i in 1:length(x), j in 1:length(x)
    z[i,j] = sdf(x[i],x[j])
end 
p = Plots.contour(x,x,z',aspect_ratio=:equal, levels=[-1,0.,1],legend=false)
# Plots.plot!(p,cps[1,:],cps[2,:],markers=:o,legend=false)
xs = reduce(hcat,square.(range(0,1,20),1))
# Plots.plot!(p,xs[1,:],xs[2,:],lw=4,marker=:x,legend=false)
display(p)

# Use mem=CUDA
# using CUDA; @assert CUDA.functional()
# using Adapt
# Adapt.adapt_structure(to, x::BSplineCurve) = BSplineCurve(adapt(to,x.cps),adapt(to,x.knots),x.count,x.degree)
# body = ParametricBody(square, (0,1); T=Float32, mem=CUDA.CuArray) # doesn't work.
# @assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
# @assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])
"""
    NurbsCurve(pnts,knos,weights)
Define a non-uniform rational B-spline curve.
- `pnts`: A 2D array representing the control points of the NURBS curve
- `knots`: A 1D array of th knot vector of the NURBS curve
- `wgts`: A 1D array of the wight of the pnts of the NURBS curve 
- `d`: The degree of the NURBS curve
"""
struct NurbsCurve{d,A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    pnts::A
    knots::V
    wgts::W
end
using StaticArrays
function NurbsCurve(pnts,knots,weights;degree=3)
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    @assert count == length(weights) "Invalid NURBS: each control point should have a corresponding weight."
    @assert count < length(knots) "Invalid NURBS: the number of knots should be greater than the number of control points."
    degree = length(knots) - count - 1 # the one in the input is not used
    NurbsCurve{degree,typeof(pnts),typeof(knots),typeof(SA{T}[weights...])}(pnts,SA{T}[knots...],SA{T}[weights...])
end
"""
    BSplineCurve(pnts; degree=3)

Define a uniform B-spline curve.
- `pnts`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
Note: An open, unifirm knot vector for a degree `degree` B-spline is constructed by default.
"""
function BSplineCurve(pnts;degree=3)
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    NurbsCurve{degree,typeof(pnts),typeof(knots),typeof(SA{T}[ones(count)...])}(pnts,knots,SA{T}[ones(count)...])
end
"""
    (::NurbsCurve)(s,t)

Evaluate the NURBS curve
- `s` position along the spline
- `t` time is currently unused but needed for ParametricBodies
"""
function (l::NurbsCurve{d})(u::T,t) where {T,d}
    pt = SA{T}[0, 0]; wsum=T(0.0)
    for k in 1:size(l.pnts, 2)
        l.knots[k]>u && break
        l.knots[k+d+1]≥u && (prod = Bd(l.knots,u,k,Val(d))*l.wgts[k];
                             pt +=prod*l.pnts[:,k]; wsum+=prod)
    end
    pt/wsum
end
"""
    Compute the Cox-De Boor recursion for B-spline basis functions.
"""
Bd(knots, u, k, ::Val{0}) = Int(knots[k]≤u<knots[k+1] || u==knots[k+1]==1)
function Bd(knots, u, k, ::Val{d}) where d
    ((u-knots[k])/max(eps(Float32),knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1))
    +(knots[k+d+1]-u)/max(eps(Float32),knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
end
# 
square = BSplineCurve(SA[5 5 0 -5 -5 -5  0  5 5
                         0 5 5  5  0 -5 -5 -5 0],degree=1)
@assert square(1f0,0) ≈ square(0f0,0) ≈ [5,0]
@assert square(.5f0,0) ≈ [-5,0]

# Does it work with KernelAbstractions?
# using CUDA; @assert CUDA.functional()
# using KernelAbstractions
# @kernel function _test!(a::AbstractArray,l::NurbsCurve)
#     # Map index to physical space
#     I = @index(Global)
#     s = (I-1)/(length(a)-1)
#     q = l(s,0)
#     a[I] = q'*q
# end
# test!(a,l)=_test!(get_backend(a),64)(a,l,ndrange=length(a))
# a = CUDA.zeros(64)
# test!(a,square)
# a|>Array # yes!

# Check bspline is type stable. Note that using Float64 for cps will break this!
@assert isa(square(0.,0),SVector)
@assert all([eltype(square(zero(T),0))==T for T in (Float32,Float64)])

# check winding direction
using LinearAlgebra
cross(a,b) = det([a;;b])
@assert all([cross(square(s,0.),square(s+0.1,0.))>0 for s in range(0,.9,10)])

# check derivatives
using ForwardDiff
dcurve(u) = ForwardDiff.derivative(u->square(u,0),u)
@assert dcurve(0f0) ≈ [0,40]
@assert dcurve(0.5f0) ≈ [0,-40]

# Wrap the shape function inside the parametric body class and check measurements
using Pkg
Pkg.activate("/home/marin/Workspace/ParametricBodies.jl")
using ParametricBodies
body = ParametricBody(square, (0,1));
@assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
@assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])

# Check that the locator works for closed splines
@assert [body.locate(SA_F32[5,s],0) for s ∈ (-2,-1,-0.1)]≈[0.95,0.975,0.9975]

# Does it work with ParametricBodies on CUDA?
# body = ParametricBody(square, (0,1); T=Float32);
# CUDA.@allowscalar @assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
# CUDA.@allowscalar @assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]]) # yes!

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
body = ParametricBody(circle, (0,1));
@assert all(measure(body,[-6,0],0) .≈ [1,[-1,0],[0,0]])
@assert all(measure(body,[ 5,5],0) .≈ [√(5^2+5^2)-5,[√2/2,√2/2],[0,0]])
@assert all(measure(body,[-5,5],0) .≈ [√(5^2+5^2)-5,[-√2/2,√2/2],[0,0]])

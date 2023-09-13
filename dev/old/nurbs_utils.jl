
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
    BSplineCurve(pnts; degree=3, mem=Array)

Define a B-spline curve.
- `pnts`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
- `mem`: Array memory type
"""
struct BSplineCurve{A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    pnts::A
    knots::V
    weights::W
    count::Int
    degree::Int
end
function BSplineCurve(pnts;degree=3) 
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    knots = [zeros(T,degree); collect(T,range(0, count-degree) / (count-degree)); ones(T,degree)]
    wgts = ones(T,count)
    BSplineCurve(pnts,SA[knots...],SA[wgts...],count,degree)
end
function NURBSCurve(pnts,wgts,knot)
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    degree = length(knot)-count-1
    BSplineCurve(copy(pnts),SA{T}[knot...],SA{T}[wgts...],count,degree)
end
function (l::BSplineCurve)(s::T,t)::SVector where {T} # `t` is currently unused
    @assert 0 ≤ s ≤ 1 "Parameter `s` must be in the range [0,1]"
    pt = zero(l.pnts[:,1]); wsum=T(0.0)
    for k in range(0, l.count-1)
        prod = coxDeBoor(l.knots, s, k, l.degree, l.count) * l.weights[k+1]
        pt += prod * l.pnts[:, k+1]
        wsum += prod
    end
    return pt/wsum
end

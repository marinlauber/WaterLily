using Plots

function coxDeBoor(knots, u, k, d, count)
    if (d == 0)
        return Int(((knots[k+1] <= u) && (u < knots[k+2])) || ((u >= (1.0-1e-12)) && (k == (count-1))))
    end
    return (((u-knots[k+1])/max(1e-12, knots[k+d+1]-knots[k+1]))*coxDeBoor(knots, u, k, (d-1), count)
          + ((knots[k+d+2]-u)/max(1e-12, knots[k+d+2]-knots[k+2]))*coxDeBoor(knots, u, (k+1), (d-1), count))
end

function bspline(cv, s; d=3)
    count = size(cv, 2)
    knots = vcat(zeros(d), range(0, count-d) / (count-d), ones(d))
    pt = zeros(size(cv, 1))
    for k in range(0, count-1)
        pt += coxDeBoor(knots, s, k, d, count) * cv[:, k+1]
    end
    return pt
end

function FindSpan(knots, u, n, d)
    if (u==knots[n+1])
        return u
    end
    low = d; high=n+1
    mid = div(low+high,2)
    while (u<knots[mid]) || (u>=knots[mid+1])
        if (u<knots[mid])
            high = mid
        else
            low = mid
        end
        mid = div(low+high,2)
    end
    return mid
end

function Basis_ITS0(knots, u, k, p)
    N = ones(p); L = zeros(p); R = zeros(p)
    for j ∈ 1:p
        saved = 0.0
        L[j] = u - knots[k+1-j]
        R[j] = knots[k+j] - u
        for r ∈ 1:j
            temp = N[r]/(R[r]+L[j-r+1])
            N[r] = saved + R[r]*temp
            saved = L[j-r+1]*temp
        end
        N[j] = saved
    end
    return N
end

function bspline_inverted_triangular_scheme(cv, s; d=3)
    count = size(cv, 2)
    knots = vcat(zeros(d), range(0, count-d) / (count-d), ones(d))
    pt = zeros(size(cv, 1))
    for k in range(0, count-1)
        pt += Basis_ITS0(knots, s, k, d, count) * cv[:, k+1]
    end
    return pt
end


function evaluate_spline(cps, s, d)
    return hcat([bspline(cps, u, d=d) for u in s]...)
end

# make sim
p=3
Points = 3L.+0.5*L*Array(reduce(hcat,[SVector(i-5, 3sin(i^2)) for i in 1:9])')
Points = Array(Points')

# Define B-spline
Xs = evaluate_spline(Points, 0:0.01:1, 3)
xs = Xs[1,:]; ys=Xs[2,:]

Plots.plot(xs,ys,color=:black,marker=:x,lw=1,legend=false)
Plots.plot!(Points[1,:],Points[2,:],markers=:o,legend=false)


# test find FindSpan
knots = [0,0,0,1,2,3,4,4,5,5,5]
u = 5/2
p=2
@assert 5==FindSpan(knots, u, length(knots)-p-1, p)

Basis_ITS0(knots, u, 6, 2)
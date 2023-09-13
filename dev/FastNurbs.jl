using BenchmarkTools
using BasicBSpline
using StaticArrays
using Plots
using LinearAlgebra: norm
using Roots

# function B(i,p,k,t)
#     if p==0
#         return float(k[i]≤t≤k[i+1])
#     else
#         return B(i,p-1,k,t)*(t-k[i])/(k[i+p]-k[i]) + B(i+1,p-1,k,t)*(k[i+p+1]-t)/(k[i+p+1]-k[i+1])
#     end
# end


# knotvector = sort(rand(10))
# p = 3
# i = 1
# t = 0.2
# @benchmark B($i,$p,$knotvector,$t)

# k = KnotVector(knotvector)
# P = BSplineSpace{p}(k)
# @benchmark bsplinebasis($P,$i,$t)

# @benchmark bsplinebasisall($P,$i,$t)

# ## 1-dim B-spline manifold
# p = 3 # degree of polynomial
# a = [SVector(i-5, 3*sin(i^2)) for i in 1:9] # control points
# # number of knot is npts + p + 1, with p+1 repeated for exact interpolation
# k = KnotVector([repeat([1],p);1:size(a,1)-p+1;repeat([size(a,1)-p+1],p)]) # knot vector
# P = BSplineSpace{p}(k) # B-spline space
# # dP = BSplineDerivativeSpace{1}(P)
# dp = p-1
# dP = BSplineSpace{dp}(KnotVector([repeat([1],dp);1:size(a,1)-dp+1;repeat([size(a,1)-dp+1],dp)]))
# M = BSplineManifold(a, P) # Define B-spline manifold
# dM = BSplineManifold(a, dP) # Define B-spline manifold
# plot(M)
# plot(dM)
# @benchmark BSplineManifold($a,$P)

# # Derivative and support
# bsplinebasisall( P,i,t)
# bsplinebasisall(dP,i,t)
# bsplinesupport(dP,5)
# plot(P)
# plot!(BSplineDerivativeSpace{1}(P))

# manual construct b-spline
p = 5
a = [SVector(i-5, 3*sin(i^2)) for i in 1:9] # control points
n = size(a,1)-p
k = KnotVector(0:1/n:1) + p*KnotVector([0,1])
P = BSplineSpace{p}(k)
dP=BSplineDerivativeSpace{1}(P)

function Scurve(a,P,t)
    i = intervalindex(P,t)
    b = bsplinebasisall(P,i,t)
    v = b[1]*getindex(a,i)
    for j in 1:degree(P)
        v += b[1+j]*getindex(a,i+j)
    end
    return v
end
function Sdcurve(a,dP,t)
    i = intervalindex(dP,t)
    db = bsplinebasisall(dP,i,t)
    dv = db[1]*getindex(a,i)
    for j in 1:degree(dP)+1
        dv += db[1+j]*getindex(a,i+j)
    end
    return dv/norm(dv)
end

# t = 0.001 
# plot(BSplineManifold(a,P))
# v = curve(a,P,t)
# dv = dcurve(a,dP,t); dv /= norm(dv)
# plot!([v[1]],[v[2]],marker=:o,legend=:none)
# plot!([v[1],v[1]+dv[1]],[v[2],v[2]+dv[2]],marker=:>,legend=:none)


# function
curve(t) = Scurve(a,P,t)
dcurve(t) = Sdcurve(a,dP,t)

# Signed distance to the curve with t ∈ [0,1]
# candidates(X) = union(0,1,find_zeros(t -> (X-curve(t))'*dcurve(t),0,1,naive=true,no_pts=3,xatol=0.01))
# function distance(X,t)
#     V = X-curve(t)
#     √(V'*V)
# end
# sdfnurbs(X) = X -> argmin(abs, distance(X,t) for t in candidates(X))

# X=[-5,2]
# # X = [-1,2]
# # X = [0.,0.1]
# cand = candidates(X)
# t = cand[argmin(distance(X,t) for t in cand)]
# plot(BSplineManifold(a,P))
# v = curve(a,P,t)
# dv = dcurve(a,dP,t); dv /= norm(dv)
# plot!([v[1]],[v[2]],marker=:o,legend=:none)
# plot!([X[1]],[X[2]],marker=:s,legend=:none)
# plot!([v[1],v[1]+dv[1]],[v[2],v[2]+dv[2]],marker=:>,legend=:none)
# savefig("FastNURBS2.png")

# using ForwardDiff

# # is forward diff faster?
# x = 0.5
# @benchmark de = dcurve(x)
# @benchmark dc = ForwardDiff.derivative(x->curve(x), x)
# @benchmark fd = (curve(x+1e-6)-curve(x))/1e-6
# display(de)
# display(fd/norm(fd))
# display(dc/norm(dc))


# # Better distance function
# pow2(x) = sum(x.*x)


# # coarse scan for t ∈ [0,1]
# N = 10
# ts = collect(0:1/N:1)
# candidate = [pow2(X-curve(t)) for t ∈ ts]
# idt = argmin(candidate)
# ti = ts[idt]

# d = distance(X,t)

# function test()
#     ts=[0,1];N=16;Δd=1;d=10; k=0
#     while Δd > 0.1 && k < N
#         println(ts[1]," ",ts[2])
#         candidate = [pow2(X-curve(t)) for t ∈ ts[1]:1/N:ts[2]]
#         ts .= (sortperm(candidate)[1:2].-1)/N
#         Δd=d-distance(X,ts[1]);k+=1
#     end
#     println(ts)
# end


function get_interval(f,df,X,t₋=0,t₊=1,N=200)
    # c = [(X-f(t))'*df(t) for t ∈ range(t₋,t₊,N+1)]
    c = [(X-f(t))'*(X-f(t)) for t ∈ range(t₋,t₊,N+1)]
    idx=sortperm(abs.(c))[1:2]; tₐ,tᵇ=range(t₋,t₊,N+1)[idx]
    d=min(norm(X-f(tₐ)),norm(X-f(tᵇ)))
    d,tₐ,tᵇ
end

function closest_point(X,f,df)
    Δd=10;tₐ=0;tᵇ=1;k=1;dⁱ=10;N=200
    while Δd > 0.01 && k < 100
        dᵏ,tₐ,tᵇ = get_interval(f,df,X,tₐ,tᵇ,N÷k)
        Δd=abs(dᵏ-dⁱ)/dⁱ;k+=1;dⁱ=dᵏ
    end
    return dⁱ,(tₐ+tᵇ)/2
end

z = [exp(im*θ) for θ ∈ range(0,2π,64)]

anim = @animate for e in range(1,100)
    X = [10,4].*rand(2).-[5,2]
    d,t = closest_point(X,curve,dcurve)
    println("X: ", X)
    println("d: ",d," t: ",t)
    plot(BSplineManifold(a,P))
    v = Scurve(a,P,t)
    plot!([X[1]],[X[2]],marker=:s,legend=:none)
    plot!([v[1]],[v[2]],marker=:o,legend=:none)
    plot!(real(d.*z).+X[1],imag(d.*z).+X[2],lw=0.5,legend=:none,aspect_ratio=:equal)
end
gif(anim, "anim_fps15.gif", fps = 2)
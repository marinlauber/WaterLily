# using StaticArrays
using Roots
using LinearAlgebra: norm

@fastmath function bernstein_B(u::T, deg::Integer) where T
    #calculate the 1st Bernstein polynomials
    B = zeros(T,1,deg+1)
    B[:, 1] = ones(T,1)
    u1 = ones(T,1) .- u
    u2 = ones(T,1) .+ u
    for j in 1:deg
        saved = zeros(T,1)
        for k in 1:j
            temp = B[:,k]
            B[:,k] = saved + u1.*temp
            saved  = u2.*temp
        end
        B[:,j+1] = saved
    end
    return B./(2^deg)
end

@fastmath function bernstein_dB(u::T, deg::Integer) where T
    #calculate the 1st derivative of Bernstein polynomials
    dB = zeros(T,1,deg)
    dB[:,1] = ones(T,1)
    u1 = ones(T,1) .- u
    u2 = ones(T,1) .+ u
    for j in 1:deg-1
        saved = zeros(T,1)
        for k in 0:j-1
            temp = dB[:,k+1]
            dB[:,k+1] = saved + u1.*temp
            saved = u2.*temp
        end
        dB[:,j+1] = saved
    end
    dB = dB./(2^deg)
    dB = hcat(zeros(T,1,1), dB, zeros(T,1, 1))
    dB = (dB[:,1:end-1]-dB[:,2:end])*deg
    return dB
end

function nurbs_sdf(Points; deg=3)
    # Define cubic Bezier
    curve(t) = [(bernstein_B(t, deg) * Points)...]
    dcurve(t) =[(bernstein_dB(t, deg)* Points)...]

    # Signed distance to the curve
    candidates(X) = union(-1,1,find_zeros(t -> (X-curve(t))'*dcurve(t),-1,1,naive=true,no_pts=3,xatol=0.01))
    function distance(X,t)
        V = X-curve(t)
        √(V'*V)
    end
    X -> argmin(abs, distance(X,t) for t in candidates(X))
end

struct SplineBody{T,Pf<:AbstractArray{T}} <: AbstractBody
    pts :: Pf
    sdf :: Function
    map :: Function
    deg :: Integer
    function SplineBody(pts;map=(x,t)->x,thk=1.0,deg=3,compose=true,T=Float32)
        pts = convert.(T,pts)
        sdf(x) = nurbs_sdf(pts;deg=deg)([x...]) .- thk/2.
        comp(x,t) = compose ? sdf(map(x,t)) : sdf(x)
        new{T,typeof(pts)}(pts,comp,map,deg)
    end
end
"""
    d = sdf(body::SplineBody,x,t) = body.sdf(x) # time depence is lost here
"""
sdf(body::SplineBody,x,t) = body.sdf(x,t)

using ForwardDiff
"""
    d,n,V = measure(body::AutoBody,x,t)

Determine the implicit geometric properties from the `sdf` and `map`.
The gradient of `d=sdf(map(x,t))` is used to improve `d` for pseudo-sdfs.
The velocity is determined _solely_ from the optional `map` function.
"""
function measure(body::SplineBody,x,t)
    # eval d=f(x,t), and n̂ = ∇f
    d = body.sdf(x,t)
    n = ForwardDiff.gradient(x->body.sdf(x,t), x)

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # The velocity depends on the material change of ξ=m(x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x->body.map(x,t), x)
    dot = ForwardDiff.derivative(t->body.map(x,t), t)
    return (d,n,-J\dot)
end

"""
    Compute the force along the spline
""" 
function ∮ds(body::SplineBody,p::AbstractArray{T},ts=collect(T,-1:0.02:1)) where T
    x = vcat([WaterLily.bernstein_B(tᵢ, 3)*body.pts for tᵢ ∈ ts]...)
    n = vcat([WaterLily.bernstein_dB(tᵢ, 3)*body.pts for tᵢ ∈ ts]...)
    δ = Float32(2.0)
    f = zeros(T,size(x,1)-1,2)
    for i in 1:size(x,1)-1
        dx,dy = x[i+1,1]-x[i,1],x[i+1,2]-x[i,2]
        dl = √(dx^2+dy^2)
        xi = SVector((x[i+1,1]+x[i,1])/2.,(x[i+1,2]+x[i,2])/2.)
        ni = SVector((n[i+1,1]+n[i,1])/2.,(n[i+1,2]+n[i,2])/2.)
        ni = ni/norm(ni)
        p_at_X = WaterLily.interp(convert.(T,xi.+δ.*ni),p)
        p_at_X .-= WaterLily.interp(convert.(T,xi.-δ.*ni),p)
        f[i,:] = p_at_X.*dl.*ni
    end
    return f
end

"""
    test function
"""
function test_nurb()
    Points = zeros(5,2)
    Points[1,:] = [0.02,-0.25].-rand(2)./4
    Points[2,:] = [0.25,-0.06]
    Points[3,:] = [0.5,0.].-[0.,rand(1)[1]./4]
    Points[4,:] = [1.,0.].+rand(2)./4
    Points[5,:] = [1.,-0.25]

    # Define sdf
    nurbs = SplineBody(Points; deg=4)

    # plot sdf
    x = range(-0.5, 1.5, length=100)
    y = range(-0.5, 0.5, length=50)
    z = zeros(length(y),length(x))
    for i in 1:length(y), j in 1:length(x)
        z[i,j] = nurbs.sdf([x[j],y[i]],0.0)
    end 
    p = contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
    Plots.plot!(p,nurbs.pts[:,1],nurbs.pts[:,2],markers=:o,legend=false)

    # extract point to proble field
    xp = hcat([bernstein_B(t, 4)*nurbs.pts for t ∈ -1:0.05:1]'...)'
    Plots.plot!(p,xp[:,1],xp[:,2],color=:red,lw=4,legend=false)
    display(p)
end
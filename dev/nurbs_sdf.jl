using WaterLily
using Plots; gr()
using StaticArrays
using Roots
using BenchmarkTools
using LinearAlgebra

@fastmath function bernstein_B(u::Number, deg::Integer)
    #calculate the 1st Bernstein polynomials
    B = zeros(1, deg+1)
    B[:, 1] = ones(1)
    u1 = ones(1) .- u
    u2 = ones(1) .+ u
    for j in 1:deg
        saved = zeros(1)
        for k in 1:j
            temp = B[:,k]
            B[:,k] = saved + u1.*temp
            saved  = u2.*temp
        end
        B[:,j+1] = saved
    end
    return B./(2^deg)
end
@fastmath function bernstein_dB(u::Number, deg::Integer)
    #calculate the 1st derivative of Bernstein polynomials
    dB = zeros(1, deg)
    dB[:,1] = ones(1)
    u1 = ones(1) .- u
    u2 = ones(1) .+ u
    for j in 1:deg-1
        saved = zeros(1)
        for k in 0:j-1
            temp = dB[:,k+1]
            dB[:,k+1] = saved + u1.*temp
            saved = u2.*temp
        end
        dB[:,j+1] = saved
    end
    dB = dB./(2^deg)
    dB = hcat(zeros(1,1), dB, zeros(1, 1))
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

function test()
    Points = zeros(5,2)
    Points[1,:] = [0.02,-0.25].-rand(2)./4
    Points[2,:] = [0.25,-0.06]
    Points[3,:] = [0.5,0.].-[0.,rand(1)[1]./4]
    Points[4,:] = [1.,0.].+rand(2)./4
    Points[5,:] = [1.,-0.25]

    # Define sdf
    nurbs(x,y) = nurbs_sdf(Points; deg=4)([x,y])
    # @benchmark nurbs(1,1)

    x = range(-0.5, 1.5, length=100)
    y = range(-0.5, 0.5, length=50)
    z = nurbs.(x',y)
    p = contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
    Plots.plot!(p, Points[:,1],Points[:,2],markers=:o,legend=false)

    # extract point to proble field
    xp = hcat([bernstein_B(t, 4)*Points for t ∈ -1:0.05:1]'...)'
    Plots.plot!(p, xp[:,1],xp[:,2],color=:red,lw=4,legend=false)
    display(p)
end


function ∮ds(xs,n,p)
    x,y = xs[:,1],xs[:,2]
    δ = 2.0
    f = zeros(length(x)-1,2)
    for i in 1:length(x)-1
        dx,dy = x[i+1]-x[i],y[i+1]-y[i]
        dl = √(dx^2+dy^2)
        xi = SVector((x[i+1]+x[i])/2.,(y[i+1]+y[i])/2.)
        ni = SVector((n[i+1,1]+n[i,1])/2.,(n[i+1,2]+n[i,2])/2.)
        ni = ni/norm(ni)
        p_at_X = WaterLily.interp(xi.+δ.*ni,p)
        f[i,:] = p_at_X.*dl.*ni
    end
    return f
end

function test_line()
    Nd = (20,20,2)
    Points = zeros(4,2)
    δx = (-4+4*√2.)/3.
    Points[1,:] = [5.,15]
    Points[2,:] = [5+10*δx,15.]
    Points[3,:] = [15,5+10*δx]
    Points[4,:] = [15.,5]

    # extract point to proble field
    xp = vcat([bernstein_B(t, 3)*Points for t ∈ -1:0.02:1]...)
    normal = vcat([bernstein_dB(t, 3)*Points for t ∈ -1:0.02:1]...)
    pressure = Array{Float64}(undef, Nd[1:end-1]...);
    pλ(x) = x[1]
    apply!(pλ,pressure); WaterLily.BC!(pressure);
    println(" p = ")
    display(pressure)

    x,y = xp[:,1],xp[:,2]
    normal = ones(length(x),2);
    normal[:,1].=sin.(range(0,π/2,length(x)));
    normal[:,2].=cos.(range(0,π/2,length(x)))
    p = Plots.contourf(pressure,color=:RdBu_11,lw=0.,levels=31,aspect_ratio=:equal)
    for i in 1:length(x)-1
        dx = [x[i+1],x[i]]
        dy = [y[i+1],y[i]]
        Plots.plot!(p,dx,dy,legend=:none)
        xi,yi = sum(dx)/2.,sum(dy)/2.
        Plots.plot!(p,[xi],[yi],marker=:o,legend=:none)
        ni = (normal[i+1,:]+normal[i,:])/2.
        ni = ni/norm(ni)
        Plots.plot!(p,[xi,xi+ni[1]],[yi,yi+ni[2]],legend=:none)
    end
    display(p)
    force = 4.0.*∮ds(xp,normal,pressure)./(π*(11)^2)
    println(" force = ")
    println(sum(force,dims=1))
end

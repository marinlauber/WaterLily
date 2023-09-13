# using WaterLily
using Plots; gr()
using StaticArrays
using Roots
using BenchmarkTools

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

# a "random" mesh
Points = zeros(5,2)
Points[1,:] = [0.02,-0.25].-rand(2)./4
Points[2,:] = [0.25,-0.06]
Points[3,:] = [0.5,0.].-[0.,rand(1)[1]./4]
Points[4,:] = [1.,0.].+rand(2)./4
Points[5,:] = [1.3,-0.25]

# Define sdf to NURBS
sdf(x,y) = nurbs_sdf(Points; deg=4)([x,y])
@benchmark sdf(1,1)

# plot
x = range(-0.5, 1.5, length=100)
y = range(-0.5, 0.5, length=50)
z = sdf.(x',y)
p = contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
Plots.plot!(p, Points[:,1],Points[:,2],markers=:o,legend=false)

# extract point to proble field
xp = hcat([bernstein_B(t, 4)*Points for t ∈ -1:0.05:1]'...)'
Plots.plot!(p, xp[:,1],xp[:,2],color=:red,lw=4,legend=false)
display(p)

# make a pressure field

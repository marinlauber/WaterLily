using Plots
using Roots

function bezier_sdf(P0,P1,P2,P3)
    # Define cubic Bezier
    A,B,C,D = P0,-3P0+3P1,3P0-6P1+3P2,-P0+3P1-3P2+P3
    curve(t) = A+B*t+C*t^2+D*t^3
    dcurve(t) = B+2C*t+3D*t^2

    # Signed distance to the curve
    candidates(X) = union(0,1,find_zeros(t -> (X-curve(t))'*dcurve(t),0,1,naive=true,no_pts=3,xatol=0.01))
    function distance(X,t)
        V = X-curve(t)
        copysign(√(V'*V),[V[2],-V[1]]'*dcurve(t))
    end
    X -> argmin(abs, distance(X,t) for t in candidates(X))
end

function test()
    sdf(x,y) = bezier_sdf([0,0],[0.5,0.],[1.0,0.5],[1.,1.])([x,y])
    x = range(-0.5, 1.5, length=100)
    y = range(-0.5, 1.5, length=50)
    z = sdf.(x',y)
    contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
    # savefig(".\\figures\\distance function.png")
end

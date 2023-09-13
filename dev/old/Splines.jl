using WaterLily
using ParametricBodies
using StaticArrays
using Plots

# parameters
L=2^4
Re=250
U =1

# NURBS points, weights and knot vector for a circle
cps = SA[1 1 0 -1 -1 -1  0  1 1
         0 1 1  1  0 -1 -1 -1 0]
cps_s = cps.*[L/2,L/10] .+ [L,3L/2]
cps_m = MMatrix(cps_s)
weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

# other
# cps = SA[-1 0 1
#          0 0 0]
# cps_s = cps.*[L/2,L/2] .+ [L,3L/2]
# cps_m = MMatrix(cps_s)
# weights = SA[1.,1.,1.]
# knots =   SA[0,0,0,1,1,1.]

# # overload the distance function
# ParametricBodies.dis(p,n) = √(p'*p) - 2.
# Body.surf.pnts .= SA[-1 0 1
                    #  -1 0 0].*[L/2,L/2].+SA[L,L]

# make a nurbs curve
circle = NurbsCurve(copy(cps_m),knots,weights)

# make a body and a simulation
# Body = ParametricBody(circle,(0,1))
Body = DynamicBody(circle,(0,1))
sim = Simulation((2L,2L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# measure the body
p1 = Plots.contourf(sim.flow.μ₀[:,:,1]',levels=1,cmap=:RdBu)
Plots.plot!(p1,Body.surf.pnts[1,:],Body.surf.pnts[2,:],c=:black,marker=:circle,legend=:none)

# update position manualy
Body.surf.pnts .= cps.*[L/2,L/2].+SA[L,2L/3]
ParametricBodies.update!(Body.locate,Body.surf,0,range(0,1,64))
measure!(sim,0)
p2 = Plots.contourf(sim.flow.μ₀[:,:,1]',levels=1,cmap=:RdBu)
Plots.plot!(p2,Body.surf.pnts[1,:],Body.surf.pnts[2,:],c=:black,marker=:circle,legend=:none)
Plots.plot(p1,p2)

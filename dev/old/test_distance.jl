using WaterLily
using Plots

# range and storage
x = range(-50, 50, length=100)
y = range(-50, 50, length=100)
z = zeros(length(y),length(x))


## vertical line
deg = 2
pts = hcat([0.,1.],[0.5,0.],[0.,-1])*32
weights = [1.,1.,1.0]
knots = WaterLily.KnotVec(pts, deg)

# plot sdf
sdf(x,y) = WaterLily.nurbs_sdf(pts,weights,knots,deg)([x,y])
for i in 1:length(y), j in 1:length(x)
    z[i,j] = sdf(x[j],y[i])
end
p = Plots.contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
Plots.plot!(p,pts[1,:],pts[2,:],markers=:o,legend=false)
display(p)


## circle
deg = 2
pts = hcat([0.,-1.],[1.,-1.],[ 1.,0.],[1.,1.],
            [0.,1. ],[-1.,1.],[-1.,0.],[-1.,-1.],
            [0.,-1.])*10
weights = [1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots = [0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1]

# plot sdf
sdf(x,y) = WaterLily.nurbs_sdf(pts,weights,knots,deg)([x,y])
for i in 1:length(y), j in 1:length(x)
    z[i,j] = sdf(x[j],y[i])
end 
p = Plots.contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
Plots.plot!(p,pts[1,:],pts[2,:],markers=:o,legend=false)
display(p)


## funny curve
deg = 4
pts = reduce(hcat,[[i-5, 3sin(i^2)] for i in 1:9]).*10
knots = WaterLily.KnotVec(pts, deg)
weights = ones(size(pts,2))

# plot sdf
sdf(x,y) = WaterLily.nurbs_sdf(pts,weights,knots,deg)([x,y])
for i in 1:length(y), j in 1:length(x)
    z[i,j] = sdf(x[j],y[i])
end 
p = Plots.contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
Plots.plot!(p,pts[1,:],pts[2,:],markers=:o,legend=false)
display(p)

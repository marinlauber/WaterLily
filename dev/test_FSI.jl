using WaterLily
using StaticArrays
using LinearAlgebra
using Plots

function ∮ds(z,n,p)
    x,y = real(z),imag(z)
    δ = 2.0
    pressure = zeros(length(x)-1,2)
    dls = zeros(length(x)-1)
    for i in 1:length(x)-1
        dx,dy = x[i+1]-x[i],y[i+1]-y[i]
        dl = √(dx^2+dy^2)
        dls[i] = dl
        xi = SVector((x[i+1]+x[i])/2.,(y[i+1]+y[i])/2.)
        ni = (n[i+1]+n[i])/2.
        ni = ni/norm(ni)
        ni = SVector(real(ni),imag(ni))
        p_at_X = WaterLily.interp(xi.+δ.*ni,p)
        pressure[i,:] = p_at_X.*dl.*ni
    end
    return pressure,dls
end

# Nd = (10,10,10,3)
Nd = (10,10,2);
# Nd = (10,10)
u = Array{Float64}(undef, Nd...);
p = Array{Float64}(undef, Nd[1:end-1]...);
uλ(i,x) = x[1];
pλ(x) = x[1];
apply!(uλ,u);
apply!(pλ,p);
println(" u = ")
display(u)
display(p)

x = SVector(4.0,2.0)
display(WaterLily.interp(x,u))
display(WaterLily.interp(x,p))

m,n=2^7,2^7
radius = m/4
Re = 250
center = n/2

body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
z = [radius*exp(im*θ) for θ ∈ range(0,2π,length=m)]
normal = z
z = z.+(center+center*im)
sim = Simulation((n,m), (0,0), radius; ν=radius/Re, body, T=Float64)

# hydrostatic pressure
pλ(x) = x[2]
apply!(pλ, sim.flow.p)

# pressure force
force = WaterLily.∮nds(sim.flow.p,sim.flow.V,sim.body)/(π*(radius)^2)
println(" force = ")
display(force)

# other pressure force
pressure, dl = ∮ds(z,normal,sim.flow.p)./(π*(radius)^2)
println(" pressure = ")
println(sum(pressure,dims=1))

# check the probe points
x,y=real(z),imag(z)
p=Plots.contourf(sim.flow.p,color=:RdBu_11,lw=0.,levels=31,aspect_ratio=:equal)
for i in 1:length(x)-1
    dx = [x[i+1],x[i]]
    dy = [y[i+1],y[i]]
    Plots.plot!(p,dx,dy,legend=:none)
    xi,yi = sum(dx)/2.,sum(dy)/2.
    Plots.plot!(p,[xi],[yi],marker=:o,legend=:none)
    ni = (normal[i+1]+normal[i])/2.
    ni = ni/norm(ni)
    ni = 2*SVector(real(ni),imag(ni))
    Plots.plot!(p,[xi,xi+ni[1]],[yi,yi+ni[2]],legend=:none)
end
display(p)

using WaterLily
using ParametricBodies
using StaticArrays
include("examples/TwoD_plots.jl")

# velocity profile of Turek Hron
function uλ(i,xy)
    x,y = @. xy .- 2
    i!=1 && return 0.0
    ((y < 0) && (y > n-1)) && return 0.0 # correct behaviour on ghost cells
    return 1.5*U*y/(n-1)*(1.0-y/(n-1))/(0.5)^2
end

# overwrite the momentum function so that we get the correct BC
@fastmath function WaterLily.mom_step!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC_TH!(a.u,a.U)
    WaterLily.project!(a,b); BC_TH!(a.u,a.U)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC_TH!(a.u,a.U,2)
    WaterLily.project!(a,b,2); a.u ./= 2; BC_TH!(a.u,a.U)
    push!(a.Δt,WaterLily.CFL(a))
end

# BC function using the profile
function BC_TH!(a,A,f=1)
    N,n = WaterLily.size_u(a)
    for j ∈ 1:n, i ∈ 1:n
        if i==j # Normal direction, impose profile on inlet and outlet
            for s ∈ (1,2,N[j])
                @WaterLily.loop a[I,i] = f*uλ(i,loc(i,I)) over I ∈ WaterLily.slice(N,s,j)
            end
        else  # Tangential directions, interpolate ghost cell to homogeneous Dirichlet
            @WaterLily.loop a[I,i] = -a[I+δ(j,I),i] over I ∈ WaterLily.slice(N,1,j)
            @WaterLily.loop a[I,i] = -a[I-δ(j,I),i] over I ∈ WaterLily.slice(N,N[j],j)
        end
    end
end

function TurekHron(x,t)
    min(√sum(abs2, x .- center) - D/2,
        √sum(abs2, x .- SA[clamp(x[1],2.5D,6D),center[2]]) - D/5)
end

function cylinder(x,t)
    √sum(abs2, x .- center) - D/2
end

# simulation parameters
U = 1
Re=200
D = 32
ϵ=0.5
thk=2ϵ+√2
m,n = 11D,4D
center = [2D,2D]

# define a flat plat at and angle of attack
cps = SA[0 1.75 3.5
         0 0 0]*D .+ [2.5D,2D]

# make a nurbs curve
flap = BSplineCurve(MMatrix(cps);degree=2)

body = AutoBody(TurekHron)
# body = AutoBody(cylinder) + DynamicBody(flap,(0,1);dist=(p,n)->√(p'*p)-thk/2)
sim = Simulation((m,n), (U,0), D; ν=U*D/Re, body, uλ=uλ)

duration=10.0
step=0.1
t₀ = round(sim_time(sim))
@time @gif for tᵢ in range(t₀,t₀+duration;step)
    sim_step!(sim,tᵢ)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[inside(sim.flow.p)],clims=(-10,10))
    # flood(sim.flow.u[inside(sim.flow.p),1],clims=(0,1.5))
    body_plot!(sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
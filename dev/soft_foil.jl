using WaterLily,Plots,StaticArrays
include("bezier_sdf.jl")
export bezier_sdf
include("nurbs_sdf.jl")
export bernstein_B,bernstein_dB

function nurbs_sdf(Points; deg=3, thk=2ϵ+√2)
    # Define cubic Bezier
    curve(t) = [(bernstein_B(t, deg) * Points)...]
    dcurve(t) =[(bernstein_dB(t, deg)* Points)...]

    # Signed distance to the curve
    candidates(X) = union(-1,1,find_zeros(t -> (X-curve(t))'*dcurve(t),-1,1,naive=true,no_pts=3,xatol=0.01))
    function distance(X,t)
        V = X-curve(t)
        # copysign(√(V'*V),[V[2],-V[1]]'*dcurve(t))
        √(V'*V)
    end
    X -> argmin(abs, distance(X,t) for t in candidates(X))
end

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(ω',dpi=300,
    color=palette(:BuGn), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)

function foil(L;d=0,Re=38e3,U=1,n=6,m=3)   
   
end

# dimensions
L = 32
U = 1
Re = 38e3
n,m = 6,3
d=0

# Pitching motion around the pivot
ω = 2π*U/L # reduced frequency k=π
center = SA[1,m/2] # foil placement in domain
pivot = SA[.1,0] # pitch location from center
function map(x,t)
    α = 6π/180*cos(ω*t)
    SA[cos(α) sin(α); -sin(α) cos(α)]*(x/L-center-pivot) + pivot
end

# Define sdf to symmetric and deflected foil
# symmetric = bezier_sdf(SA[0,0],SA[0,0.1],SA[0.5,0.12],SA[1.,0.])
symmetric = nurbs_sdf(Array([[0,0] [0,0.1] [0.5,0.12] [1.,0.]]');deg=3)
deflect(x) = max(0,x-0.3)^2/0.7^2
sdf(x,time) = L*symmetric(SA[x[1],abs(x[2]+d*deflect(x[1]))])

# foil sim
sim = Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body=AutoBody(sdf,map))
#
# # get start time
# t₀ = round(sim_time(sim))
# duration = 15; step = 0.1;

# @time @gif for tᵢ in range(t₀,t₀+duration;step)

#     # update until time tᵢ in the background
#     sim_step!(sim,tᵢ)

#     # flood plot
#     get_omega!(sim);
#     plot_vorticity(sim.flow.σ, limit=10)

#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
using WaterLily,StaticArrays
using ParametricBodies # New package
function make_sim(;L=32,Re=1e3,St=0.3,αₘ=-π/18,U=1,n=8,m=4,T=Float32,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    nose,pivot = SA[L,0.5f0m*L],SA[0.25f0L,0]
    θ₀ = T(αₘ+atan(π*St)); h₀=T(L); ω=T(π*St*U/h₀)
    function map(x,t)
        back = x[1]>nose[1]+2L       # back body?
        ϕ = back ? 5.5f0 : 0         # phase shift
        S = back ? 3L : 0            # horizontal shift
        θ = θ₀*cos(ω*t+ϕ); R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        h = SA[S,h₀*sin(ω*t+ϕ)]
        ξ = R*(x-nose-h-pivot)+pivot # move to origin and align with x-axis
        return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
    end

    # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]
    body = ParametricBody(foil,(0,1);map,T,mem)

    Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body,T,mem)
end

sim = make_sim();

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1

# step and plot
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ; @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(a[inside(a)],clims=(-10,10)); body_plot!(sim)
end
using WaterLily
using ForwardDiff
using StaticArrays

struct PorousBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1
    map::F2
    κ:: Float64
    ε:: Float64
    function PorousBody(κ, ε, sdf, map=(x,t)->x; compose=true)
        comp(x,t) = compose ? sdf(map(x,t),t) : sdf(x,t)
        new{typeof(comp),typeof(map)}(comp, map, κ, ε)
    end
end
WaterLily.sdf(body::PorousBody,x,t;kwargs...) = body.sdf(x,t)
function WaterLily.measure(body::PorousBody,x,t;fastd²=Inf)
    # eval d=f(x,t), and n̂ = ∇f
    d = body.sdf(x,t)
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    n = ForwardDiff.gradient(x->body.sdf(x,t), x)
    any(isnan.(n)) && return (d,zero(x),zero(x))

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # The velocity depends on the material change of ξ=m(x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x->body.map(x,t), x)
    dot = ForwardDiff.derivative(t->body.map(x,t), t)
    return (d,n,-J\dot)
end
import WaterLily: inside,@loop,size_u,slice,CI,δ,ϕ,ϕu,ϕuL,ϕuR,∂,inside_u
function WaterLily.measure!(a::Flow{N,T},body::PorousBody;t=zero(T),ϵ=1) where {N,T}
    # https://pdf.sciencedirectassets.com/271422/1-s2.0-S0167610518X00063/1-s2.0-S016761051730692X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIF3kBoWpYyrMveadhHYP8Z%2F3LMah6l8I6feW5mHfFRSeAiEAlyPmPnEx3uR%2BGOvC6HrZ9D3IztLIibhPlaQow57YqgMquwUIjv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDE2Ny08RszYBGDkG6iqPBWpjlk7BWde5FJIk2RaQx0yGzoCwrwVCZdQYS6h7%2FdrpHNwoPtPMNkPQc4oDP2GV%2BaqAuq6iS6HT0VAK8Gf38mqVFgo4G66rTBUKWyxbtUGJAitrXVl5OIKrFqQFO5ObX1N2LlYX1ke2JWqL%2Be8iE6u7ZhXeVoUfb5xPs5B0SXkjS2ph1THHIklPSzxr6Yl8VKTvwLXTlEnM9kE3ExnjxdXJC2lNa6ay6F2y5S97paG%2F2KIxjUO2BgR6SmxBX73EpAUapLOKmTZylX1X4%2BF07RabtVdZFWI2grCI8BVox9QL9mdVD7Vp857f6dV4hoKGwIOdZeEfh1PJnXRiWWUPYz6wNj60luVmwPk6aQQCYPnPBn7ln2Vv8HvHg4Fx9joczbLxp6PB5NKBhk%2FIgnpwA55CJSL0X9EZT2Ml81vnRB0U4R8xD1OBbYLlsPwpHHxjiIzN7%2Bvr1AW%2BulrBn3%2BP1CEoykVOHhyCNTB7EkdFnqBDipkGQu2uYLizhRy2qz0GbUZWhw2S6uXuz1NgRcieOsOPUR5Z4s5CIm%2BaZYue084sS57pOAF2KkguTGiuRUpk2VMrLQNP%2FPzZ7PgxMxhBs780vPzgTg9Y%2Bt1kceOafQ1THZO%2Fq7NYycPQVFqZ4KLo%2FZA97WBfm%2BBPiJou8niOOIdzDsW1hG5p0CfmUvKWEu%2FO%2Fqta8ORUTcx1oLvk5rMdXjDBJ2CUN7DVmLbVY6WvX7PUfmfhWnzlRkJk5zmQslVaqTzRgtdSpRmTjTiAmpQLlxrt%2BrXQvYFF6cDGHnsC8yuc%2FcgrjtjHpO8kQ9p7RYEG280UWAWHzMtW87lUOwvpNinONILDK2JB7CyHwsgZet3lq877iWKTV4orS7dMa34w8LaztQY6sQFpPjmSkEk6Ds9KKiimDHxqagzFdbKZHnw22NgoCUNbN5IOR9SteUP2tE46iU3FAz9jFy%2BQlqIZxMpzDZ7V85DzY8wU%2FpIR4hutI%2B6du64EyWWOIMAaR3NeRFtiWDownomh1%2FaWqtZhD2zU7UB4YuYwFlzuiFnVVAi9h74a%2F9VjU9v4fWG7gNaKVTCyJj3R4fjRFzwrEHXedzm2xsTlrRVG778kwshTtM7NDcHVRt3PVKw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240802T142824Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY73LGLZRG%2F20240802%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c745aa5b33e0ad28fb01c8963d15337b5b572246457aea6bca7ebfa9bc553434&hash=3aea9facafb36406594b69c06823165b8846b170ffbf608f8c1fe51398a09fdf&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S016761051730692X&tid=spdf-0e984a96-fffc-4986-b218-ff703a57056f&sid=f7e6054284c4284fd1599b94495e5200c3a9gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=08085b06060500565555&rr=8acec1144ed706de&cc=nl
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T); d²=(2+ϵ)^2
    @fastmath @inline function fill!(μ₀,μ₁,V,d,I)
        d[I] = sdf(body,loc(0,I,T),t,fastd²=d²)
        if d[I]^2<d²
            for i ∈ 1:N
                dᵢ,nᵢ,Vᵢ = measure(body,loc(i,I,T),t,fastd²=d²)
                V[I,i] = Vᵢ[i]
                μ₀[I,i] = WaterLily.μ₀(dᵢ,ϵ)
                # for j ∈ 1:N
                #     μ₁[I,i,j] = WaterLily.μ₁(dᵢ,ϵ)*nᵢ[j]
                # end
            end
        elseif d[I]<zero(T)
            for i ∈ 1:N
                μ₀[I,i] = zero(T)
            end
        end
    end
    @loop fill!(a.μ₀,a.μ₁,a.V,a.σ,I) over I ∈ inside(a.p)
    BC!(a.μ₀,zeros(SVector{N,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
    BC!(a.V ,zeros(SVector{N,T}),a.exitBC,a.perdir)
    # now that we have filled the kernel, we compute the Velocity V from the 
    # Darcy-Brinkman-Forchheimer equation
    DarcyBrinkmanForchheimer!(a.V,a.u,a.σ;ν=a.ν,κ=body.κ,ε=body.ε)
end
function DarcyBrinkmanForchheimer!(r,u,Φ;ν=0.1,κ=0.1,ε=1)
    F = 1.75/√150/ε^1.5 # inertial factor (Dhinakaran and Ponmozhi, 2011).
    r .= 0.
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # treatment for bottom boundary with BCs
        @loop r[I,i] += (1/ε^2)*ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν/ε*∂(j,CI(I,i),u) - (ν/κ+F/√κ*√sum(abs2,(u[I,:])))*u[I,i] over I ∈ slice(N,2,j,2)
        # inner cells
        @loop (Φ[I] = (1/ε^2)*ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν/ε*∂(j,CI(I,i),u) - (ν/κ+F/√κ*√sum(abs2,(u[I,:])))*u[I,i];
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        @loop r[I-δ(j,I),i] += -(1/ε^2)*ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u)) + ν/ε*∂(j,CI(I,i),u) - (ν/κ+F/√κ*√sum(abs2,(u[I,:])))*u[I,i] over I ∈ slice(N,N[j],j,2)
    end
end

using WaterLily
include("../../Tutorials-WaterLily/src/TwoD_plots.jl")

function circle(n,m;Re=250,U=1)
    radius, center = m/8, m/2
    body = PorousBody(1e6,10,(x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body)
end

# Initialize and run
sim = circle(3*2^6,2^7)
sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true,remeasure=true)


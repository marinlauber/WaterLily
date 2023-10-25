using LinearAlgebra
using StaticArrays
using Plots
using IterativeSolvers
using BenchmarkTools

L₂(x) = sqrt(sum(abs2,x))/length(x)

function backsub(A,b)
    n = size(A,1)
    x = zeros(n)
    x[n] = b[n]/A[n,n]
    for i in n-1:-1:1
        s = sum( A[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / A[i,i]
    end
    return x
end

abstract type AbstractCoupling end

struct Relaxation <: AbstractCoupling
    ω :: Float64                  # relaxation parameter
    x :: AbstractArray{Float64}   # primary variable
    function Relaxation(x⁰::AbstractArray{Float64};relax::Float64=0.5)
        new(relax,copy(x⁰))
    end
end
function update(cp::Relaxation, xᵏ) 
    # store variable and residual
    rᵏ = xᵏ .- cp.x
    # relaxation update
    xᵏ .= cp.x .+ cp.ω*rᵏ; cp.x .= xᵏ
    return xᵏ
end

struct IQNCoupling <: AbstractCoupling
    ω :: Float64
    x :: AbstractArray{Float64}
    x̃ :: AbstractArray{Float64}
    r :: AbstractArray{Float64}
    V :: AbstractArray{Float64}
    W :: AbstractArray{Float64}
    iter :: Vector{Int64}
    function IQNCoupling(x⁰::AbstractVector{Float64};ω::Float64=0.5)
        N = length(x⁰)
        new(ω,copy(x⁰),zeros(N),zeros(N),zeros(N,N),zeros(N,N),[0])
    end
end
function update(cp::IQNCoupling, xᵏ)
    if cp.iter[1]==0
        # store variable and residual
        rᵏ = xᵏ .- cp.x; cp.x̃.=xᵏ
        # relaxation update
        xᵏ .= cp.x .+ cp.ω*rᵏ
        # store
        cp.x.=xᵏ; cp.r.=rᵏ
        cp.iter[1] = cp.iter[1] + 1
    else
        # residuals
        rᵏ = xᵏ .- cp.x
        # roll the matrix to make space for new column
        roll!(cp.V); roll!(cp.W)
        cp.V[:,1] = rᵏ .- cp.r; cp.r .= rᵏ
        cp.W[:,1] = xᵏ .- cp.x̃; cp.x̃ .= xᵏ # save old solver iter
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view cp.V[:,1:min(cp.iter[1],N)])
        cᵏ = backsub(Rᵏ,-Qᵏ'*rᵏ)
        xᵏ.= cp.x .+ (@view cp.W[:,1:min(cp.iter[1],N)])*cᵏ .+ rᵏ #not sure
        # update for next step
        cp.x .= xᵏ
        cp.iter[1] = cp.iter[1] + 1
    end
    return xᵏ
end
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])


function Relaxation(x0, H, k=0)
    ω = 0.05; resid = []; rᵏ=1
    while L₂(rᵏ) > 1e-10 && k < 1000
        xᵏ = H(x0)
        rᵏ = xᵏ .- x0
        x0 = x0 .+ ω.*rᵏ
        k+=1;
        push!(resid,L₂(rᵏ))
    end
    return x0,resid
end

# non-symmetric matrix wih know eigenvalues
N = 20
λ = 10 .+ (1:N)
# A = triu(rand(N,N),1) + diagm(λ)
A = rand(N,N) + diagm(λ)
b = rand(N);

# IQNILS method requires a fixed point
H(x) = x + (b - A*x)

x0 = copy(b)
sol,history = IterativeSolvers.gmres(A,b;log=true, reltol=1e-16)
r3 = history.data[:resnorm]

p = plot(r3, marker=:s, xaxis=:log10, yaxis=:log10, label="IterativeSolvers.gmres",
         xlabel="Iteration", ylabel="Residual",
         xlim=(1,200), ylim=(1e-16,1e2), legend=:bottomleft)

x0 = copy(b)
@time sol,resid_relax = Relaxation(x0,H, 0.05)
# @assert L₂(f(sol)) < 1e-6
plot!(p, resid_relax, marker=:o, xaxis=:log10, yaxis=:log10, label="Relaxation",
      legend=:bottomleft)

# QN couling
# IQNSolver = IQNCoupling(zero(x0);ω=0.05)
IQNSolver = IQNCoupling(copy(x0);ω=0.05)

k=1; resid=[]; sol=[]; rᵏ=1.0
@time while L₂(rᵏ) > 1e-16 && k < 2N
    global x0, rᵏ, k, resid, sol
    # fsi uperator
    xᵏ = H(x0)
    # compute update
    x0 = update(IQNSolver, xᵏ)
    push!(sol,xᵏ)
    rᵏ = IQNSolver.r
    push!(resid,L₂(rᵏ))
    k+=1
end

plot!(p, resid, marker=:o, xaxis=:log10, yaxis=:log10, label="IQN-ILS struct",
      legend=:bottomleft)
# savefig(p, "GMRESvsIQNILS.png")
p


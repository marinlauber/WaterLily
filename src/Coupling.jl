using LinearAlgebra
using StaticArrays
using Plots
using IterativeSolvers

L₂(x) = sqrt(sum(abs2,x))/length(x)

abstract type AbstractCoupling end

struct ConstantRelaxation <: AbstractCoupling
    ω :: Float64
    r :: AbstractArray{Float64}
    x :: AbstractArray{Float64}
    function ConstantRelaxation(N::Int64;ω::Float64=0.5)
        new(ω,zeros(N),zeros(N))
    end
end
function update(cp::ConstantRelaxation, xᵏ, rᵏ)
    x = (1-cp.ω)*xᵏ .+ cp.ω*cp.x; cp.x .= xᵏ
    r = (1-cp.ω)*rᵏ .+ cp.ω*cp.r; cp.r .= rᵏ
    return x, r
end

struct IQNCoupling <: AbstractCoupling
    ω :: Float64
    r :: AbstractArray{Float64}
    x :: AbstractArray{Float64}
    V :: AbstractArray{Float64}
    W :: AbstractArray{Float64}
    iter :: Vector{Int64}
    function IQNCoupling(N::Int64;ω::Float64=0.5)
        new(ω,zeros(N),zeros(N),zeros(N,N),zeros(N,N),[0])
    end
end
function update(cp::IQNCoupling, xᵏ, rᵏ)
    if cp.iter[1]==0
        # store variable and residual
        cp.x .= xᵏ; cp.r .= rᵏ
        # relaxation update
        xᵏ .+= cp.ω*rᵏ
        cp.iter[1] = 1
    else
        # roll the matrix to make space for new column
        roll!(cp.V); roll!(cp.W)
        cp.V[:,1] = rᵏ .- cp.r; cp.r .= rᵏ
        cp.W[:,1] = xᵏ .- cp.x; cp.x .= xᵏ
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view cp.V[:,1:min(cp.iter[1],N)])
        cᵏ = backsub(Rᵏ,-Qᵏ'*rᵏ)
        xᵏ.+= (@view cp.W[:,1:min(cp.iter[1],N)])*cᵏ #.+ rᵏ #not sure
        cp.iter[1] = cp.iter[1] + 1
    end
    return xᵏ
end
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
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])



# non-symmetric matrix wih know eigenvalues
N = 100
λ = 10 .+ (1:N)
A = rand(N,N) + diagm(λ)
b = rand(N);

# IQNILS method
f(x) = b - A*x

# Reference solution
x0 = copy(b)
@time sol,history = IterativeSolvers.gmres(A,b;log=true, reltol=1e-16)
resid = history.data[:resnorm]
@assert L₂(f(sol)) < 1e-6

p = plot(resid, marker=:d, xaxis=:log10, yaxis=:log10, label="IterativeSolvers.GMRES",
         xlabel="Iteration", ylabel="Residual",
         xlim=(1,200), ylim=(1e-16,1e2), legend=:bottomleft)

# IQNILS method
IQNSolver = IQNCoupling(N;ω=0.05)
xᵏ = copy(b); rᵏ = f(xᵏ); k=1; resid=[]; sol=[]
@time while L₂(rᵏ) > 1e-16 && k < 2N
    global xᵏ, rᵏ, k, resid, sol
    xᵏ = update(IQNSolver, xᵏ, rᵏ)
    rᵏ = f(xᵏ)
    push!(sol,xᵏ)
    push!(resid,L₂(rᵏ))
    k+=1
end
@assert L₂(f(sol[end])) < 1e-6

plot!(p, resid, marker=:o, xaxis=:log10, yaxis=:log10, label="IQN-ILS struct", legend=:bottomleft)
# savefig(p, "GMRESvsIQNILS.png")
p
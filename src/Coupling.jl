using LinearAlgebra: norm,qr,Diagonal

abstract type AbstractCoupling end

struct Relaxation <: AbstractCoupling
    ω :: Float64                  # relaxation parameter
    x :: AbstractArray{Float64}   # primary variable
    xₛ :: AbstractArray{Float64}  # secondary variable
    function Relaxation(primary::AbstractArray{Float64},secondary::AbstractArray{Float64};relax::Float64=0.5)
        new(relax,zero(primary),zero(secondary))
    end
end
function update(cp::Relaxation, x_new, x_newₛ) 
    # relax primary data
    r = x_new .- cp.x
    cp.x .= x_new
    x_new .+= cp.ω.*r
    # relax secondary data
    rₛ = x_newₛ .- cp.xₛ
    cp.xₛ .= x_newₛ
    x_newₛ .+= cp.ω.*rₛ
    return x_new, x_newₛ
end

struct IQNCoupling <: AbstractCoupling
    ω :: Float64                    # intial relaxation
    x :: AbstractArray{Float64}     # primary variable
    r :: AbstractArray{Float64}     # primary residual
    xₛ :: AbstractArray{Float64}   # secondary variabe
    rₛ :: AbstractArray{Float64}   # secondary residual
    V :: AbstractArray{Float64}     # primary residual difference
    W :: AbstractArray{Float64}     # primary variable difference
    Wₛ :: AbstractArray{Float64}    # secondary variable difference
    iter :: Vector{Int64}           # iteration counter
    function IQNCoupling(primary::AbstractArray{Float64},secondary::AbstractArray{Float64};relax::Float64=0.5)
        N1=length(primary); N2=length(secondary)
        # Wₛ is of size (N2,N1) as there are only N1 cᵏ
        new(relax,zero(primary),zero(primary),zero(secondary),zero(secondary),
            zeros(N1,N1),zeros(N1,N1),zeros(N2,N1),[0])
    end
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
# function backsub!(x::AbstractVector,A::AbstractArray,b::AbstractVector)
#     n = size(A,1)
#     x[n] = b[n]/A[n,n]
#     for i in n-1:-1:1
#         s = sum( A[i,j]*x[j] for j in i+1:n )
#         x[i] = ( b[i] - s ) / A[i,i]
#     end
# end
function update(cp::IQNCoupling, x_new, x_newₛ)
    if cp.iter[1]==0 # relaxation step
        # relax primary data
        r = x_new .- cp.x
        cp.x .= x_new
        cp.r .= r
        x_new .+= cp.ω.*r
        # relax secondary data
        rₛ = x_newₛ .- cp.xₛ
        cp.xₛ .= x_newₛ
        cp.rₛ .= rₛ
        x_newₛ .+= cp.ω.*rₛ
        cp.iter[1] = 1 # triggers QN update
    else
        k = cp.iter[1]; N = length(cp.x)
        # compute residuals
        r = x_new .- cp.x
        rₛ= x_newₛ.- cp.xₛ
        # roll the matrix to make space for new column
        roll!(cp.V); roll!(cp.W); roll!(cp.Wₛ)
        cp.V[:,1] = r .- cp.r;      cp.r .= r
        cp.W[:,1] = x_new .- cp.x;  cp.x .= x_new
        cp.Wₛ[:,1]= x_newₛ.- cp.xₛ; cp.xₛ.= x_newₛ # secondary data
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view cp.V[:,1:min(k,N)])
        cᵏ = backsub(Rᵏ,-Qᵏ'*r)
        println(size(cᵏ))
        x_new  .+= (@view cp.W[:,1:min(k,N)]) *cᵏ .+ r # not sure
        x_newₛ .+= (@view cp.Wₛ[:,1:min(k,N)])*cᵏ .+ rₛ # secondary data
        cp.iter[1] = k + 1
    end
    return x_new, x_newₛ
end
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])
# pop!(A::AbstractArray,k) = (A[:,k:end-1] .= A[:,k+1:end]; A[:,end].=0)
# relative resudials
res(a,b) = norm(a-b)/norm(b)




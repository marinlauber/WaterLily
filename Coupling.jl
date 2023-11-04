using LinearAlgebra: norm,dot

struct CoupledSimulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    stru
    function Simulation(dims::NTuple{N}, u_BC::NTuple{N}, L::Number;
                        Δt=0.25, ν=0., U=√sum(abs2,u_BC), ϵ=1,
                        uλ::Function=(i,x)->u_BC[i],
                        body::AbstractBody=NoBody(),T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,T,f=mem)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ))
    end
end
function sim_step!(sim::CoupledSimulation,t_end;verbose=false,remeasure=true)
    t = time(sim)
    while t < t_end*sim.L/sim.U
        store!(sim); iter=1
        while iter < 50
            # update structure
            solve_step!(sim.struc,force,t,sim.flow.Δt[end])
            # update flow
            ParametricBodies.update!(sim.body,pnts,sim.flow.Δt[end])
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            force=force(sim.body,sim); pnts=points(sim.struc)
            # check convergence and accelerate
            update!(sim.coupling,force,pnts,Val(iter==1)) && break
            # revert if not convergend
            revert!(sim); iter+=1
        end
        #update time
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function store(sim::CoupledSimulation)
    sim.uˢ .= sim.flow.u; sim.pˢ .= sim.flow.p
    sim.cache = sim.struc.d, sim.struc.v, sim.struc.a
end
function revert!(sim::CoupledSimulation)
    sim.flow.u .= sim.uˢ; sim.flow.p .= sim.pˢ
    sim.struc.d, sim.struc.v, sim.struc.a .= sim.cache
end

function backsub(A,b)
    n = size(A,2)
    x = zeros(n)
    x[n] = b[n]/A[n,n]
    for i in n-1:-1:1
        s = sum( A[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / A[i,i]
    end
    return x
end

function concatenate!(vec, a, b, subs)
    vec[subs[1]] = a[1,:]; vec[subs[2]] = a[2,:];
    vec[subs[3]] = b[1,:]; vec[subs[4]] = b[2,:];
end
function revert!(vec, a, b, subs)
    a[1,:] = vec[subs[1]]; a[2,:] = vec[subs[2]];
    b[1,:] = vec[subs[3]]; b[2,:] = vec[subs[4]];
end

mutable struct QRFactorization{T}
    Q::Matrix{T}
    R::Matrix{T}
    cols::Int # how many are use actually
    rows::Int
end

function QRFactorization(V::AbstractMatrix{T},singularityLimit::T) where T
    QR = QRFactorization(zero(V), zero(V), 0, 0);
    delIndices=[];
    for i in 1:size(V,2)
        inserted = insertColumn!(QR, QR.cols+1, V[:,i], singularityLimit)
        if !inserted
            push!(delIndices, i)
        end
    end
    return QR.Q, QR.R, delIndices
end

function apply!(QR::QRFactorization, V, W, _col; singularityLimit::T=1e-6) where T
    delIndices=[];
    for i in 1:_col
        # recomputing the QR factorization every time
        # println("Inserting column ", i, " of ", size(QR.Q, 2)," with norm of Vⱼ: ", norm(V[:,i]))
        inserted = insertColumn!(QR, QR.cols+1, V[:,i], singularityLimit)
        if !inserted
            push!(delIndices, i)
        end
    end
    # pop the column that are filtered out, backward to avoid index shifting
    for k in sort(delIndices,rev=true)
        popCol!(V,k); popCol!(W,k);
    end
    return nothing
end

function insertColumn!(QR::QRFactorization{T}, k::Int, vec::Vector{T}, singularityLimit::T) where T
    # copy to avoid overwriting
    v = copy(vec)

    if QR.cols == 0
        QR.rows = length(v)
    end
    applyFilter = singularityLimit > zero(T)

    # we add a column
    QR.cols += 1

    # orthogonalize v to columns of Q
    u = zeros(T, QR.cols)
    rho_orth = zero(T)
    rho0 = zero(T)

    if applyFilter
        rho0 = norm(v)
    end

    # try to orthogonalize the new vector
    err, rho_orth = orthogonalize(QR.Q, v, u, QR.cols-1)
    
    if rho_orth <= eps(T) || err < 0
        # println("The ratio ||v_orth|| / ||v|| is extremely small and either the orthogonalization process of column v failed or the system is quadratic.")
        QR.cols -= 1
        return false
    end

    if applyFilter && (rho0 * singularityLimit > rho_orth)
        # println("Discarding column as it is filtered out by the QR2-filter: rho0 * eps > rho_orth: ", rho0 * singularityLimit, " > ", rho_orth)
        QR.cols -= 1
        return false
    end

    # populate new column and row with zeros, they exist, they are just not shown
    QR.R[:, QR.cols] .= zeros(T)
    QR.R[QR.cols, :] .= zeros(T)

    # Shift columns to the right
    for j in QR.cols - 1:-1:k
        for i in 1:j
            QR.R[i, j + 1] = QR.R[i, j]
        end
    end
    # reset diagonal
    for j in k + 1:QR.cols
        QR.R[j, j] = zero(T)
    end
    
    # add to the right
    QR.Q[:, QR.cols] = v

    # Maintain decomposition and orthogonalization by application of Givens rotations
    for l in QR.cols-2:-1:k
        s,g = computeReflector(u[l], u[l + 1])
        Rr1 = QR.R[l, 1:QR.cols]
        Rr2 = QR.R[l + 1, 1:QR.cols]
        applyReflector!(s, g, l + 1, QR.cols, Rr1, Rr2)
        QR.R[l, 1:QR.cols] = Rr1
        QR.R[l + 1, 1:QR.cols] = Rr2
        Qc1 = QR.Q[:, l]
        Qc2 = QR.Q[:, l + 1]
        applyReflector!(s, g, 1, QR.rows, Qc1, Qc2)
        QR.Q[:, l] = Qc1
        QR.Q[:, l + 1] = Qc2
    end

    for i in 1:k
        QR.R[i, k] = u[i]
    end

    return true
end


function orthogonalize(Q::AbstractMatrix{T}, v::AbstractVector{T}, r::AbstractVector{T}, colNum::Int) where T
    null = false
    termination = false
    k = 0

    r .= zeros(T)
    s = zeros(T, colNum)
    rho = norm(v)  # l2norm
    rho0 = rho; rho1 = rho

    while !termination
        u = zeros(T, length(v))
        for j in 1:colNum
            Qc = Q[:, j]
            r_ij = dot(Qc, v)
            s[j] = r_ij
            u .+= Qc * r_ij
        end
        
        for j in 1:colNum
            r[j] += s[j]
        end

        v .-= u
        rho1 = norm(v)  # Distributed l2norm
        norm_coefficients = norm(s)  # Distributed l2norm
        k += 1

        if size(Q, 2) == colNum
            # println("The least-squares system matrix is quadratic, i.e., the new column cannot be orthogonalized (and thus inserted) to the LS-system.\nOld columns need to be removed.")
            v .= zeros(T)
            rho = zero(T)
            return k, rho
        end

        if rho1 <= eps(T)
            # println("The norm of v_orthogonal is almost zero, i.e., failed to orthogonalize column v; discard.")
            null = true
            rho1 = one(T)
            termination = true
        end

        if rho1 <= rho0 * norm_coefficients
            if k >= 4
                # println("Matrix Q is not sufficiently orthogonal. Failed to reorthogonalize new column after 4 iterations. New column will be discarded. The least-squares system is very badly conditioned, and the quasi-Newton will most probably fail to converge.")
                return -1, rho1
            end
            rho0 = rho1
        else
            termination = true
        end
    end
    v ./= rho1
    rho = null ? zero(T) : rho1
    r[colNum+1] = rho
    return k, rho
end

function computeReflector(x::T, y::T) where T
    u = x
    v = y
    if v == zero(T)
        sigma = zero(T)
        gamma = one(T)
    else
        mu = max(abs(u), abs(v))
        t = mu * sqrt((u / mu)^2 + (v / mu)^2)
        t *= (u < zero(T)) ? -one(T) : one(T)
        gamma = u / t
        sigma = v / t
        x = t
        y = zero(T)
    end
    return sigma, gamma
end
function applyReflector!(sigma, gamma, k::Int, l::Int, p::Vector{T}, q::Vector{T}) where T
    nu = sigma / (one(T) + gamma)
    for j in k:l-1
        u = p[j]
        v = q[j]
        t = u * gamma + v * sigma
        p[j] = t
        q[j] = (t + u) * nu - v
    end
end

struct ResidualSum
    λ :: AbstractArray{Float64}
    w :: AbstractArray{Float64}
    iw :: AbstractArray{Float64}
    function ResidualSum(N)
        new(zeros(N),ones(N),ones(N))
    end
end
# reset the summation
update!(pr::ResidualSum,r,svec,reset::Val{true}) = pr.λ .= 0;
# update the summation
function update!(pr::ResidualSum,r,svec,reset::Val{false})
    for s in svec
        println(" preconditioner scaling factor ",norm(r)/norm(r[s]))
        pr.λ[s] .+= norm(r[s])/norm(r)
    end
    pr.w .= 1.0./pr.λ
    pr.iw .= pr.λ
    return nothing
end

abstract type AbstractCoupling end

struct Relaxation <: AbstractCoupling
    ω :: Float64                  # relaxation parameter
    x :: AbstractArray{Float64}   # primary variable
    r :: AbstractArray{Float64}   # primary variable
    subs
    function Relaxation(primary::AbstractArray{Float64},secondary::AbstractArray;relax::Float64=0.5)
        n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂
        subs = (1:m₁,m₁+1:n₁*m₁,n₁*m₁+1:n₁*m₁+m₂,n₁*m₁+m₂+1:N)
        x⁰ = zeros(N); concatenate!(x⁰,primary,secondary,subs)
        new(relax,copy(x⁰),zero(x⁰),subs)
    end
end
function update(cp::Relaxation, xᵏ, reset) 
    # store variable and residual
    rᵏ = xᵏ .- cp.x
    # relaxation updates
    xᵏ .= cp.x .+ cp.ω*rᵏ
    # xᵏ .= cp.x .- ((xᵏ.-cp.x)'*(rᵏ.-cp.r)/((rᵏ.-cp.r)'*(rᵏ.-cp.r)).-1.0)*rᵏ
    cp.x .= xᵏ; cp.r .= rᵏ
    return xᵏ
end


struct IQNCoupling <: AbstractCoupling
    ω :: Float64                    # intial relaxation
    x :: AbstractArray{Float64}     # primary variable
    x̃ :: AbstractArray{Float64}     # old solver iter (not relaxed)
    r :: AbstractArray{Float64}     # primary residual
    V :: AbstractArray{Float64}     # primary residual difference
    W :: AbstractArray{Float64}     # primary variable difference
    c :: AbstractArray{Float64}     # least-square coefficients
    P :: ResidualSum                # preconditionner
    QR :: QRFactorization{Float64}  # QR factorization
    subs                            # sub residual indices
    svec
    iter :: Dict{Symbol,Int64}      # iteration counter
    function IQNCoupling(primary::AbstractArray{Float64},secondary::AbstractArray;relax::Float64=0.5)
        n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂
        subs = (1:m₁,m₁+1:n₁*m₁,n₁*m₁+1:n₁*m₁+m₂,n₁*m₁+m₂+1:N)
        x⁰ = zeros(N); concatenate!(x⁰,primary,secondary,subs)
        svec = (1:n₁*m₁,n₁*m₁+1:N)
        new(relax,x⁰,zeros(N),zeros(N),zeros(N,N÷2),zeros(N,N÷2),zeros(N÷2),
            ResidualSum(N),
            QRFactorization(zeros(N,N÷2),zeros(N÷2,N÷2),0,0),
            subs,svec,Dict(:k=>0))
    end
end
function update(cp::IQNCoupling, xᵏ, new_ts)
    if cp.iter[:k]==0
        # compute residual and store variable
        cp.r .= xᵏ .- cp.x; cp.x̃.=xᵏ
        # relaxation update
        xᵏ .= cp.x .+ cp.ω*cp.r
        # store values
        cp.x .= xᵏ
    else
        # residuals
        rᵏ = xᵏ .- cp.x;
        # roll the matrix to make space for new column
        roll!(cp.V); roll!(cp.W)
        cp.V[:,1] = rᵏ .- cp.r; cp.r .= rᵏ
        cp.W[:,1] = xᵏ .- cp.x̃; cp.x̃ .= xᵏ # save old solver iter
        # residual sum preconditioner
        # update!(cp.P,rᵏ,cp.svec,Val(new_ts))
        cp.W .*= cp.P.w
        # QR decomposition and filter columns
        k = min(cp.iter[:k],size(cp.V,2))
        apply!(cp.QR, cp.V, cp.W, k)
        cp.V .*= cp.P.iw # revert scaling
        # solve least-square problem 
        R = @view cp.QR.R[1:k,1:k]
        Q = @view cp.QR.Q[:,1:k]
        cᵏ = backsub(R,-Q'*(cp.P.w.*rᵏ)); cp.c[1:length(cᵏ)] .= cᵏ
        Δx = (@view cp.W[:,1:length(cᵏ)])*cᵏ
        # update for next step
        xᵏ.= cp.x .+ Δx .+ rᵏ; cp.x .= xᵏ
    end
    cp.iter[:k] += 1
    return xᵏ
end
popCol!(A::AbstractArray,k) = (A[:,k:end-1] .= A[:,k+1:end]; A[:,end].=0)
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])
res(xᵏ,xᵏ⁺¹) = norm(xᵏ-xᵏ⁺¹)/norm(xᵏ)

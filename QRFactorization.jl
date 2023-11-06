using LinearAlgebra: norm,dot

"""
    Residual sum preconditionner
"""
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

"""
    QR factorization
"""
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
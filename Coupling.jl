using LinearAlgebra: norm,dot
include("QRFactorization.jl")

function writetxt(string, v::Vector{Float64})
    open(string*".txt","w") do io
        for i in 1:length(v)
            println(io, v[i])
        end
    end
end
function writetxt(string, a::Matrix{Float64})
    open(string*".txt","w") do io
        for i in 1:size(a,1)
            println(io, a[i,:])
        end
    end
end

# these are not really needed
function concatenate!(vec, a, b, subs)
    vec[subs[1]] = a[1,:]; vec[subs[2]] = a[2,:];
    vec[subs[3]] = b[1,:]; vec[subs[4]] = b[2,:];
end
function revert!(vec, a, b, subs)
    a[1,:] = vec[subs[1]]; a[2,:] = vec[subs[2]];
    b[1,:] = vec[subs[3]]; b[2,:] = vec[subs[4]];
end

abstract type AbstractCoupling end

struct CoupledSimulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    struc 
    cpl :: AbstractCoupling
    # coupling variables
    forces :: AbstractArray
    pnts :: AbstractArray
    # storage for iterations
    uˢ :: AbstractArray
    pˢ :: AbstractArray
    dˢ :: AbstractArray
    vˢ :: AbstractArray
    aˢ :: AbstractArray
    xˢ :: AbstractArray
    ẋˢ :: AbstractArray
    function CoupledSimulation(dims::NTuple{N}, u_BC::NTuple{N}, L::Number, body, struc, Coupling;
                               Δt=0.25, ν=0., U=√sum(abs2,u_BC), ϵ=1, ωᵣ=0.5, maxCol=100,
                               uλ::Function=(i,x)->u_BC[i],T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,T,f=mem); measure!(flow,body;ϵ)
        force_0 = zeros((2,length(uv_integration(struc.op))))
        pnts_0 = zero(body.surf.pnts)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ),struc,
            Coupling(pnts_0,force_0;relax=ωᵣ,maxCol),
            force_0,pnts_0,
            zeros(size(flow.u)),zeros(size(flow.p)),
            zeros(size(struc.u[1])),
            zeros(size(struc.u[2])),
            zeros(size(struc.u[3])),
            zeros(size(body.surf.pnts)),
            zeros(size(body.velocity.pnts)))
    end
end
function sim_step!(sim::CoupledSimulation,t_end;verbose=true,remeasure=true)
    t = sum(sim.flow.Δt[1:end-1])
    # @show t
    while t < t_end*sim.L/sim.U
        store!(sim); iter=1
        # @show t
        while true
            # update structure
            solve_step!(sim.struc,sim.forces,sim.flow.Δt[end]/sim.L)
            # update body
            ParametricBodies.update!(sim.body,u⁰+L*sim.pnts,sim.flow.Δt[end])
            # update flow
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            # compute new coupling variable
            sim.forces.=force(sim.body,sim.flow); sim.pnts.=points(sim.struc)
            # check convergence and accelerate
            verbose && print("    iteration: ",iter)
            converged = update!(sim.cpl,sim.pnts,sim.forces,0.0)
            # revert!(xᵏ,sim.pnts,sim.forces,sim.cpl.subs)
            (converged || iter+1 > 50) && break
            # revert if not convergend
            revert!(sim); iter+=1
        end
        #update time
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function store!(sim::CoupledSimulation)
    sim.uˢ .= sim.flow.u;
    sim.pˢ .= sim.flow.p;
    sim.dˢ .= sim.struc.u[1];
    sim.vˢ .= sim.struc.u[2]
    sim.aˢ .= sim.struc.u[3];
    sim.xˢ .= sim.body.surf.pnts;
    sim.ẋˢ .= sim.body.velocity.pnts;
end
function revert!(sim::CoupledSimulation)
    sim.flow.u .= sim.uˢ;
    sim.flow.p .= sim.pˢ;
    pop!(sim.flow.Δt)
    sim.struc.u[1] .= sim.dˢ;
    sim.struc.u[2] .= sim.vˢ;
    sim.struc.u[3] .= sim.aˢ;
    sim.body.surf.pnts .= sim.xˢ;
    sim.body.velocity.pnts .= sim.ẋˢ;
end

function update!(cp::AbstractCoupling,primary,secondary,kwargs)
    xᵏ=zero(cp.x); concatenate!(xᵏ,primary,secondary,cp.subs)
    converged = update!(cp,xᵏ,kwargs)
    revert!(xᵏ,primary,secondary,cp.subs)
    return converged
end

struct Relaxation <: AbstractCoupling
    ω :: Float64                  # relaxation parameter
    x :: AbstractArray{Float64}   # primary variable
    r :: AbstractArray{Float64}   # primary variable
    subs
    function Relaxation(primary::AbstractArray{Float64},secondary::AbstractArray;relax::Float64=0.5,maxCol=100)
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
function update!(cp::Relaxation, xᵏ, kwarg)
    # check convergence
    println(" r₂: ",res(cp.x, xᵏ), " converged: : ",res(cp.x, xᵏ)<1e-2)
    res(cp.x, xᵏ)<1e-2 && return true
    # store variable and residual
    cp.r .= xᵏ .- cp.x
    # relaxation updates
    xᵏ .= cp.x .+ cp.ω*cp.r; cp.x .= xᵏ
    return false
end
function finalize!(cp::Relaxation, xᵏ)
    # do nothing
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
    function IQNCoupling(primary::AbstractArray{Float64},secondary::AbstractArray;relax::Float64=0.5,maxCol::Integer=200)
        n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂
        subs = (1:m₁,m₁+1:n₁*m₁,n₁*m₁+1:n₁*m₁+m₂,n₁*m₁+m₂+1:N)
        x⁰ = zeros(N); concatenate!(x⁰,primary,secondary,subs)
        svec = (1:n₁*m₁,n₁*m₁+1:N)
        new(relax,x⁰,zeros(N),zeros(N),zeros(N,min(N÷2,maxCol)),zeros(N,min(N÷2,maxCol)),zeros(min(N÷2,maxCol)),
            ResidualSum(N),
            QRFactorization(zeros(N,min(N÷2,maxCol)),zeros(min(N÷2,maxCol),min(N÷2,maxCol)),0,0),
            subs,svec,Dict(:k=>0,:first=>1))
    end
end
function update(cp::IQNCoupling, xᵏ, _firstIter)
    Δx = zeros(size(cp.x))
    if cp.iter[:k]==0
        # compute residual and store variable
        cp.r .= xᵏ .- cp.x; cp.x̃.=xᵏ
        # relaxation update
        Δx .= cp.ω*cp.r;
        xᵏ .= cp.x .+ cp.ω*cp.r
        # store values
        cp.x .= xᵏ
    else
        # residuals
        rᵏ = xᵏ .- cp.x;
        k = min(cp.iter[:k],cp.QR.cols) # defualt we do not insert a column
        if !_firstIter # on a first iteration, we simply apply the relaxation
            @debug "updating V and W matrix"
            # roll the matrix to make space for new column
            roll!(cp.V); roll!(cp.W)
            cp.V[:,1] = rᵏ .- cp.r;
            cp.W[:,1] = xᵏ .- cp.x̃;
            k = min(cp.iter[:k],cp.QR.cols+1)
        end
        cp.r .= rᵏ; cp.x̃ .= xᵏ # save old solver iter
        # residual sum preconditioner
        update_P!(cp.P,rᵏ,cp.svec,Val(false))
        cp.V .*= cp.P.w
        # recompute QR decomposition and filter columns
        ε = _firstIter ? Float64(0.0) : 1e-2
        @debug "updating QR factorization with ε=$ε and k=$k"
        apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=ε)
        cp.V .*= cp.P.iw # revert scaling
        # solve least-square problem 
        R = @view cp.QR.R[1:cp.QR.cols,1:cp.QR.cols]
        Q = @view cp.QR.Q[:,1:cp.QR.cols]
        rᵏ .*= cp.P.w # apply preconditioer to the residuals
        cᵏ = backsub(R,-Q'*rᵏ)
        @debug "least-square coefficients: $cᵏ"
        cp.c[1:length(cᵏ)] .= cᵏ
        # rᵏ .*= cp.P.iw # revert preconditioer to the residuals
        Δx .= (@view cp.W[:,1:length(cᵏ)])*cᵏ
        # update for next step
        @debug "correction factor $Δx"
        @debug "residuals         $(cp.r)"
        xᵏ.= cp.x .+ Δx .+ cp.r; cp.x .= xᵏ
    end
    cp.iter[:k] += 1
    return xᵏ
end
function update_VW!(cp,x,r)
    roll!(cp.V); roll!(cp.W)
    cp.V[:,1] = r .- cp.r;
    cp.W[:,1] = x .- cp.x̃;
    min(cp.iter[:k],cp.QR.cols+1)
end
function update!(cp::IQNCoupling, xᵏ, kwarg)
    # compute the residuals
    rᵏ = xᵏ .- cp.x;
    println(" r₂: ",res(cp.x, xᵏ), " converged: : ",res(cp.x, xᵏ)<1e-2)
    # check convergence, if converged add this to the QR
    if res(cp.x, xᵏ)<1e-2
       # update V and W matrix
        k = update_VW!(cp,xᵏ,rᵏ)
        # apply the residual sum preconditioner, without recalculating
        cp.V .*= cp.P.w
        # QR decomposition and filter columns
        apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=0.0)
        cp.V .*= cp.P.iw # revert scaling
        # reset the preconditionner
        cp.P.residualSum .= 0; cp.iter[:first]=1
        return true
    end
    # first step is relaxation
    if cp.iter[:k]==0
        # compute residual and store variable
        cp.r .= rᵏ; cp.x̃.=xᵏ
        # relaxation update
        xᵏ .= cp.x .+ cp.ω*cp.r; cp.x .= xᵏ
    else
        k = min(cp.iter[:k],cp.QR.cols) # default we do not insert a column
        if !Bool(cp.iter[:first]) # on a first iteration, we simply apply the relaxation
            @debug "updating V and W matrix"
            k = update_VW!(cp,xᵏ,rᵏ)
        end
        cp.r .= rᵏ; cp.x̃ .= xᵏ # save old solver iter
        # residual sum preconditioner
        update_P!(cp.P,rᵏ,cp.svec,Val(false))
        # apply precondiotnner
        cp.V .*= cp.P.w;
        # recompute QR decomposition and filter columns
        ε = Bool(cp.iter[:first]) ? Float64(0.0) : 1e-2
        @debug "updating QR factorization with ε=$ε and k=$k"
        apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=ε)
        cp.V .*= cp.P.iw # revert scaling
        # solve least-square problem 
        R = @view cp.QR.R[1:cp.QR.cols,1:cp.QR.cols]
        Q = @view cp.QR.Q[:,1:cp.QR.cols]
        rᵏ .*= cp.P.w # apply preconditioer to the residuals
        # compute coefficients
        cᵏ = backsub(R,-Q'*rᵏ); cp.c[1:length(cᵏ)] .= cᵏ
        @debug "least-square coefficients: $cᵏ"
        # update for next step
        xᵏ .= cp.x .+ (@view cp.W[:,1:length(cᵏ)])*cᵏ .+ cp.r; cp.x .= xᵏ
    end
    cp.iter[:k]+=1; cp.iter[:first]=0
    return false
end
function finalize!(cp::IQNCoupling, xᵏ)
    # add the new contribution as it has not been made yet
    rᵏ = xᵏ .- cp.x;
    # roll the matrix to make space for new column
    roll!(cp.V); roll!(cp.W)
    cp.V[:,1] = rᵏ .- cp.r; cp.r .= rᵏ
    cp.W[:,1] = xᵏ .- cp.x̃; cp.x̃ .= xᵏ # save old solver iter
    # apply the residual sum preconditioner, without recalculating
    cp.V .*= cp.P.w
    # QR decomposition and filter columns
    k = min(cp.iter[:k],cp.QR.cols+1)
    apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=0.0)
    cp.V .*= cp.P.iw # revert scaling
    # reset the preconditionner
    cp.P.residualSum .= 0;
end
popCol!(A::AbstractArray,k) = (A[:,k:end-1] .= A[:,k+1:end]; A[:,end].=0)
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])
"""
    relative residual norm, bounded
"""
res(xᵏ,xᵏ⁺¹) = norm(xᵏ⁺¹-xᵏ)/norm(xᵏ⁺¹.+eps())

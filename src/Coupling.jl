using LinearAlgebra: norm,dot
include("QRFactorization.jl")

# utility function
function concatenate!(vec, a, b, subs)
    vec[subs[1]] = a[1,:]; vec[subs[2]] = a[2,:];
    vec[subs[3]] = b[1,:]; vec[subs[4]] = b[2,:];
end
function revert!(vec, a, b, subs)
    a[1,:] = vec[subs[1]]; a[2,:] = vec[subs[2]];
    b[1,:] = vec[subs[3]]; b[2,:] = vec[subs[4]];
end


abstract type AbstractCoupling end
"""
    update!(cp::AbstractCoupling,primary,secondary,kwargs)

Updates the coupling variable `cp` using the implemented couping scheme.
"""
function update!(cp::AbstractCoupling,primary,secondary,kwargs)
    xᵏ=zero(cp.x); concatenate!(xᵏ,primary,secondary,cp.subs)
    converged = update!(cp,xᵏ,kwargs)
    revert!(xᵏ,primary,secondary,cp.subs)
    return converged
end

"""
    CoupledSimulation()

A struct to hold the coupled simulation of a fluid-structure interaction problem.
"""
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
    function CoupledSimulation(dims::NTuple{N}, u_BC, L::Number, body, struc, Coupling;
                               Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, ωᵣ=0.5, maxCol=100,
                               perdir=(0,), uλ=nothing, exitBC=false, T=Float32, mem=Array) where N
        @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
        @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
        isa(u_BC,Function) && @assert all(typeof.(ntuple(i->u_BC(i,T(0)),N)).==T) "`u_BC` is not type stable"
        uλ = isnothing(uλ) ? ifelse(isa(u_BC,Function),(i,x)->u_BC(i,0.),(i,x)->u_BC[i]) : uλ
        U = isnothing(U) ? √sum(abs2,u_BC) : U # default if not specified
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC); measure!(flow,body;ϵ)
        force = zeros((2,length(uv_integration(struc))))
        Ns = size(struc.u[1]); Nn = size(body.surf.pnts)
        uˢ, pˢ = zero(flow.u) |> mem, zero(flow.p) |> mem
        dˢ, vˢ, aˢ = zeros(Ns) |> mem, zeros(Ns) |> mem, zeros(Ns) |> mem
        pnts,xˢ,ẋˢ = zeros(Nn) |> mem, zeros(Nn) |> mem, zeros(Nn) |> mem
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir),struc,
            Coupling(pnts,force;relax=ωᵣ,maxCol),force,pnts,uˢ,pˢ,dˢ,vˢ,aˢ,xˢ,ẋˢ)
    end
end

"""
    sim_time(sim::CoupledSimulation,t_end)

    
"""
function sim_step!(sim::CoupledSimulation,t_end;verbose=true,maxStep=15)
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
            (converged || iter+1 > maxStep) && break
            # revert if not convergend
            revert!(sim); iter+=1
        end
        #update time
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

"""
    store!(sim::CoupledSimulation)

Checkpoints that state of a coupled simulation for implicit coupling.
"""
function store!(sim::CoupledSimulation)
    sim.uˢ .= sim.flow.u
    sim.pˢ .= sim.flow.p
    sim.dˢ .= sim.struc.u[1]
    sim.vˢ .= sim.struc.u[2]
    sim.aˢ .= sim.struc.u[3]
    sim.xˢ .= sim.body.surf.pnts
    sim.ẋˢ .= sim.body.velocity.pnts
end

"""
    revert!(sim::CoupledSimulation)

Reverts to the previous state of a coupled simulation for implicit coupling.
"""
function revert!(sim::CoupledSimulation)
    sim.flow.u .= sim.uˢ
    sim.flow.p .= sim.pˢ
    pop!(sim.flow.Δt) 
    # pop the last two iter in the poisson solver
    pop!(sim.pois.n); pop!(sim.pois.n)
    sim.struc.u[1] .= sim.dˢ
    sim.struc.u[2] .= sim.vˢ
    sim.struc.u[3] .= sim.aˢ
    sim.body.surf.pnts .= sim.xˢ
    sim.body.velocity.pnts .= sim.ẋˢ
end


"""
    Relaxation

Standard Relaxation coupling scheme for implicit fluid-structure interaction simultations.

"""
struct Relaxation{T,Vf<:AbstractArray{T}} <: AbstractCoupling
    ω :: T    # relaxation parameter
    x :: Vf   # vector of coupling variable
    r :: Vf   # vector of rediuals
    subs :: Tuple # indices
    function Relaxation(primary::AbstractArray{T},secondary::AbstractArray{T};
                        relax::T=0.5,maxCol::Integer=100,mem=Array) where T
        n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂
        subs = (1:m₁,m₁+1:n₁*m₁,n₁*m₁+1:n₁*m₁+m₂,n₁*m₁+m₂+1:N)
        x⁰,r = zeros(N) |> mem, zeros(N) |> mem
        concatenate!(x⁰,primary,secondary,subs)
        new{T,typeof(x⁰)}(relax,x⁰,r,subs)
    end
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
finalize!(cp::Relaxation, xᵏ) = nothing

struct IQNCoupling{T,Vf<:AbstractArray{T},Mf<:AbstractArray{T}} <: AbstractCoupling
    ω :: T                 # intial relaxation
    x :: Vf                # primary variable
    x̃ :: Vf                # old solver iter (not relaxed)
    r :: Vf                # primary residual
    V :: Mf                # primary residual difference
    W :: Mf                # primary variable difference
    c :: AbstractArray{T}  # least-square coefficients
    P :: ResidualSum{T}    # preconditionner
    QR :: QRFactorization  # QR factorization
    subs :: Tuple          # sub residual indices
    svec :: Tuple
    iter :: Dict{Symbol,Int16}      # iteration counter
    function IQNCoupling(primary::AbstractArray{T},secondary::AbstractArray{T};
                         relax::T=0.5,maxCol::Integer=200,mem=Array) where T
        n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂; M = min(N÷2,maxCol)
        x⁰,x,r =      zeros(N) |> mem,     zeros(N) |> mem, zeros(N) |> mem
        V, W, c = zeros((N,M)) |> mem, zeros((N,M)) |> mem, zeros(M) |> mem
        subs = (1:m₁,m₁+1:n₁*m₁,n₁*m₁+1:n₁*m₁+m₂,n₁*m₁+m₂+1:N)
        concatenate!(x⁰,primary,secondary,subs); svec = (1:n₁*m₁,n₁*m₁+1:N)
        new{T,typeof(x⁰),typeof(V)}(relax,x⁰,x,r,V,W,c,ResidualSum(N;T=T,mem=mem),
                                    QRFactorization(V,0,0;f=mem),
                                    subs,svec,Dict(:k=>0,:first=>1))
    end
end
function update_VW!(cp,x,r)
    roll!(cp.V); roll!(cp.W)
    cp.V[:,1] = r .- cp.r;
    cp.W[:,1] = x .- cp.x̃;
    min(cp.iter[:k],cp.QR.dims[1]+1)
end
function update!(cp::IQNCoupling{T}, xᵏ, kwarg) where T
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
        k = min(cp.iter[:k],cp.QR.dims[1]) # default we do not insert a column
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
        ε = Bool(cp.iter[:first]) ? T(0.0) : 1e-2
        @debug "updating QR factorization with ε=$ε and k=$k"
        apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=ε)
        cp.V .*= cp.P.iw # revert scaling
        # solve least-square problem 
        R = @view cp.QR.R[1:cp.QR.dims[1],1:cp.QR.dims[1]]
        Q = @view cp.QR.Q[:,1:cp.QR.dims[1]]
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
    k = min(cp.iter[:k],cp.QR.dims[1]+1)
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
res(xᵏ::AbstractArray{T},xᵏ⁺¹::AbstractArray{T}) where T = norm(xᵏ⁺¹-xᵏ)/norm(xᵏ⁺¹.+eps(T))

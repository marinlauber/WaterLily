module WaterLilyMPIExt

if isdefined(Base, :get_extension)
    using MPI
else
    using ..MPI
end

using StaticArrays
using WaterLily
import WaterLily: init_mpi,me,mpi_grid,finalize_mpi
import WaterLily: BC!,perBC!,exitBC!,L₂,L∞,loc,_dot,CFL,residual!

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2

"""
    halos(dims,d)

Return the CartesianIndices of the halos in dimension `±d` of an array of size `dims`.
"""
function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (1:2) : (dims[i]-1:dims[i]) : (1:dims[i]), N))
end
"""
    buff(dims,d)

Return the CartesianIndices of the buffer in dimension `±d` of an array of size `dims`.
"""
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (3:4) : (dims[i]-3:dims[i]-2) : (1:dims[i]), N))
end

"""
    mpi_swap!(send1,recv1,send2,recv2,neighbor,comm)

This function swaps the data between two MPI processes. The data is sent from `send1` to `neighbor[1]` and received in `recv1`.
The data is sent from `send2` to `neighbor[2]` and received in `recv2`. The function is non-blocking and returns when all data 
has been sent and received. 
"""
function mpi_swap!(send1,recv1,send2,recv2,neighbor,comm)
    reqs=MPI.Request[]
    # Send to / receive from neighbor 1 in dimension d
    push!(reqs,MPI.Isend(send1,  neighbor[1], 0, comm))
    push!(reqs,MPI.Irecv!(recv1, neighbor[1], 1, comm))
    # Send to / receive from neighbor 2 in dimension d
    push!(reqs,MPI.Irecv!(recv2, neighbor[2], 0, comm))
    push!(reqs,MPI.Isend(send2,  neighbor[2], 1, comm))
    # wair for all transfer to be done
    MPI.Waitall!(reqs)
end

"""
    perBC!(a)

This function sets the boundary conditions of the array `a` using the MPI grid.
"""
perBC!(a::MPIArray, N, mpi::Bool) = for d ∈ eachindex(N)
    # get data to transfer @TODO use @views
    send1 = a[buff(N,-d)]; send2 =a[buff(N,+d)]
    recv1 = zero(send1);   recv2 = zero(send2)
    # swap 
    mpi_swap!(send1,recv1,send2,recv2,neighbors(d),mpi_grid().comm)

    # this sets the BCs
    !mpi_wall(d,1) && (a[halos(N,-d)] .= recv1) # halo swap
    !mpi_wall(d,2) && (a[halos(N,+d)] .= recv2) # halo swap
end

"""
    BC!(a)

This function sets the boundary conditions of the array `a` using the MPI grid.
"""
function BC!(a::MPIArray,A,saveexit=false,perdir=())
    N,n = WaterLily.size_u(a)
    for i ∈ 1:n, d ∈ 1:n
        # get data to transfer @TODO use @views
        send1 = a[buff(N,-d),i]; send2 = a[buff(N,+d),i]
        recv1 = zero(send1);     recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,neighbors(d),mpi_grid().comm)

        # this sets the BCs on the domain boundary and transfers the data
        if mpi_wall(d,1) # left wall
            if i==d # set flux
                a[halos(N,-d),i] .= A[i]
                a[WaterLily.slice(N,3,d),i] .= A[i]
            else # zero gradient
                a[halos(N,-d),i] .= reverse(send1; dims=d)
            end
        else # neighbor on the left
            a[halos(N,-d),i] .= recv1
        end
        if mpi_wall(d,2) # right wall
            if i==d && (!saveexit || i>1) # convection exit
                a[halos(N,+d),i] .= A[i]
            else # zero gradient
                a[halos(N,+d),i] .= reverse(send2; dims=d)
            end
        else # neighbor on the right
            a[halos(N,+d),i] .= recv2
        end
    end
end

function exitBC!(u::MPIArray,u⁰,U,Δt)
    N,_ = WaterLily.size_u(u)
    exitR = WaterLily.slice(N.-2,N[1]-2,1,3) # exit slice excluding ghosts
    # ∮udA = 0
    # if mpi_wall(1,2) #right wall
    @WaterLily.loop u[I,1] = u⁰[I,1]-U[1]*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = sum(u[exitR,1])/length(exitR)-U[1]   # mass flux imbalance
    # end
    # ∮u = MPI.Allreduce(∮udA,+,mpi_grid().comm)           # domain imbalance
    # mpi_wall(1,2) && (@WaterLily.loop u[I,1] -= ∮u over I ∈ exitR) # correct flux only on right wall
    @WaterLily.loop u[I,1] -= ∮u over I ∈ exitR # correct flux only on right wall
end

struct MPIGrid #{I,C<:MPI.Comm,N<:AbstractVector,M<:AbstractArray,G<:AbstractVector}
    me::Int                    # rank
    comm::MPI.Comm             # communicator
    coords::AbstractVector     # coordinates
    neighbors::AbstractArray   # neighbors
    global_loc::AbstractVector # the location of the lower left corner in global index space
end
const MPI_GRID_NULL = MPIGrid(-1,MPI.COMM_NULL,[-1,-1,-1],[-1 -1 -1; -1 -1 -1],[0,0,0])

let
    global MPIGrid, set_mpi_grid, mpi_grid, mpi_initialized, check_mpi

    # allows to access the global mpi grid
    _mpi_grid::MPIGrid          = MPI_GRID_NULL
    mpi_grid()::MPIGrid         = (check_mpi(); _mpi_grid::MPIGrid)
    set_mpi_grid(grid::MPIGrid) = (_mpi_grid = grid;)
    mpi_initialized()           = (_mpi_grid.comm != MPI.COMM_NULL)
    check_mpi()                 = !mpi_initialized() && error("MPI not initialized")
end

function init_mpi(Dims::NTuple{D};dims=[0, 0, 0],periods=[0, 0, 0],comm::MPI.Comm=MPI.COMM_WORLD,
                  disp::Integer=1,reorder::Bool=true) where D
    # MPI
    MPI.Init()
    nprocs = MPI.Comm_size(comm)
    # create cartesian communicator
    MPI.Dims_create!(nprocs, dims)
    comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
    me     = MPI.Comm_rank(comm_cart)
    coords = MPI.Cart_coords(comm_cart)
    # make the cart comm
    neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
    for i = 1:NDIMS_MPI
        neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
    end
    # global index coordinate in grid space
    global_loc = SVector([coords[i]*Dims[i] for i in 1:D]...)
    set_mpi_grid(MPIGrid(me,comm_cart,coords,neighbors,global_loc))
    return me; # this is the most usefull MPI vriable to have in the local space
end
finalize_mpi() = MPI.Finalize()

# global coordinate in grid space
# grid_loc(;grid=MPI_GRID_NULL) = 0
grid_loc(;grid=mpi_grid()) = grid.global_loc
me()= mpi_grid().me
neighbors(dim) = mpi_grid().neighbors[:,dim]
mpi_wall(dim,i) = mpi_grid().neighbors[i,dim]==MPI.PROC_NULL

L₂(a::MPIArray{T}) where T = MPI.Allreduce(sum(T,abs2,@inbounds(a[I]) for I ∈ inside(a)),+,mpi_grid().comm)
function L₂(p::Poisson{T,S}) where {T,S<:MPIArray{T}} # should work on the GPU
    MPI.Allreduce(sum(T,@inbounds(p.r[I]*p.r[I]) for I ∈ inside(p.r)),+,mpi_grid().comm)
end
L∞(a::MPIArray) = MPI.Allreduce(maximum(abs.(a)),Base.max,mpi_grid().comm)
L∞(p::Poisson{T,S}) where {T,S<:MPIArray{T}} = MPI.Allreduce(maximum(abs.(p.r)),Base.max,mpi_grid().comm)
function _dot(a::MPIArray{T},b::MPIArray{T}) where T
    MPI.Allreduce(sum(T,@inbounds(a[I]*b[I]) for I ∈ inside(a)),+,mpi_grid().comm)
end

function CFL(a::Flow{D,T,S};Δt_max=10) where {D,T,S<:MPIArray{T}}
    @inside a.σ[I] = WaterLily.flux_out(I,a.u)
    MPI.Allreduce(min(Δt_max,inv(maximum(a.σ)+5a.ν)),Base.min,mpi_grid().comm)
end
# this actually add a global comminutation every time residual is called
function residual!(p::Poisson{T,S}) where {T,S<:MPIArray{T}}
    WaterLily.perBC!(p.x,p.perdir)
    @inside p.r[I] = ifelse(p.iD[I]==0,0,p.z[I]-WaterLily.mult(I,p.L,p.D,p.x))
    # s = sum(p.r)/length(inside(p.r))
    s = MPI.Allreduce(sum(p.r)/length(inside(p.r)),+,mpi_grid().comm)
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I]-s
end

# function sim_step!(sim::Simulation,t_end;remeasure=true,max_steps=typemax(Int),verbose=false,mpi=true)
#     steps₀ = length(sim.flow.Δt)
#     while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
#         sim_step!(sim; remeasure)
#         (verbose && me()==0) && println("tU/L=",round(sim_time(sim),digits=4),
#                                         ", Δt=",round(sim.flow.Δt[end],digits=3))
#     end
# end

end # module
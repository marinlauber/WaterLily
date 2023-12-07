using ParametricBodies
using StaticArrays
using Splines
using CUDA

struct FlexibleBody{T,S<:Function,L<:Union{Function,NurbsLocator},V<:Function,D<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    velocity::V # v = velocity(uv), defaults to v=0
    scale::T   #|dx/dξ| = scale
    dist::D
    op # :: AbstractFEOperator
end
function FlexibleBody(surf,locate,op;dist=dis,T=Float64)
    # Check input functions
    x,t = SVector{2,T}(0,0),T(0);
    @CUDA.allowscalar uv = locate(x,t); p = x-surf(uv,t)
    @assert isa(uv,T) "locate is not type stable"
    @assert isa(p,SVector{2,T}) "surf is not type stable"
    @assert isa(dist(x,x),T) "dist is not type stable"
    dsurf = copy(surf); dsurf.pnts .= 0.0 # zero velocity
    FlexibleBody(surf,locate,dsurf,T(1.0),dist,op)
end
"""
    FlexibleBody(surf,uv_bounds;step,t⁰,T,mem,map) <: AbstractBody

Creates a `FlexibleBody` with `locate=NurbsLocator(surf,uv_bounds...)`.
"""
FlexibleBody(surf,uv_bounds::Tuple,op;dist=dis,step=1,t⁰=0.,T=Float64) =
             FlexibleBody(surf,NurbsLocator(surf,uv_bounds;step,t⁰,T,mem=Array),op;dist,T)


numElem=2
degP=3
ptLeft = 0.0
ptRight = 1.0

# parameters
EI = 0.35         # Cauhy number
EA = 100_000.0  # make inextensible
density(ξ) = 0.5  # mass ratio

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptRight, 0.0; comp=1),
    Boundary1D("Neumann", ptRight, 0.0; comp=2)
]

# make a structure
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, 
                          Dirichlet_BC, Neumann_BC; ρ=density, 
                          ρ∞=0.0)

function FlexibleBody(op::AbstractFEOperator,uv_bounds;dist=(p,n)->√(p'*p),T=Float64)
    surf = NurbsCurve(op.mesh.controlPoints,op.mesh.knots,op.mesh.weights)
    FlexibleBody(surf,NurbsLocator(surf,uv_bounds;step=1,t⁰=0.,T,mem=Array),op;dist,T)
end

# construct from mesh, this can be tidy
u⁰ = MMatrix(SA[0 0.5 1
                0 0   0])

spline = BSplineCurve(u⁰;degree=2)


body = FlexibleBody(spline,(0,1),struc;dist=(p,n)->√(p'*p))

# pnts = @view mesh.controlPoints[1:2,:]